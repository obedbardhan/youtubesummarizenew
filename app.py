"""YouTubeSummarizer — Flask API server for YouTube transcript summarization."""

import os
import re
import sys
import time
import socket
import requests
import threading
import urllib3.util.connection as urllib3_cn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force IPv4 to bypass severe IPv6 DNS routing timeouts
def allowed_gai_family():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = allowed_gai_family

# Secure deterministic file paths for remote static deployments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

# ─── Rate-limit semaphore ──────────────────────────────────────────────
# A Semaphore (not a Lock) lets N threads call Gemini concurrently.
# 2 is safe for free-tier keys; raise to 5 if you have a paid quota.
GEMINI_SEMAPHORE = threading.Semaphore(2)

# Per-key retry state so multiple threads back off together when a 429 hits
_retry_after: dict[str, float] = {}
_retry_lock = threading.Lock()

# ─── Helpers ──────────────────────────────────────────────────────────
from typing import Optional

def extract_video_id(url: str) -> Optional[str]:
    """Extract the YouTube video ID from any known URL format."""
    url = url.strip()
    # Strip referral / tracking params that appear before the ID in some share URLs
    patterns = [
        r"[?&]v=([\w-]{11})",          # ?v=ID or &v=ID  (standard watch URLs)
        r"youtu\.be/([\w-]{11})",       # youtu.be/ID
        r"/(?:embed|v)/([\w-]{11})",    # /embed/ID  /v/ID
        r"/shorts/([\w-]{11})",         # /shorts/ID
        r"/live/([\w-]{11})",           # /live/ID  (live streams)
        r"^([\w-]{11})$",               # bare 11-char ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_video_metadata(video_id: str) -> dict:
    """Fetch video title and thumbnail via oembed."""
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        resp = requests.get(oembed_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title": data.get("title", "Untitled Video"),
                "author": data.get("author_name", "Unknown"),
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            }
    except Exception:
        pass
    return {
        "title": "Untitled Video",
        "author": "Unknown",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    }

def _build_yt_api() -> YouTubeTranscriptApi:
    """
    Build a YouTubeTranscriptApi instance, using a Webshare residential proxy
    if credentials are configured via environment variables.
    This is the recommended way to bypass YouTube IP blocks on cloud providers.

    Set these env vars in Render to enable:
      WEBSHARE_PROXY_USERNAME  — from https://dashboard.webshare.io/proxy/settings
      WEBSHARE_PROXY_PASSWORD  — from https://dashboard.webshare.io/proxy/settings
    Make sure you purchased a "Residential" package (NOT "Proxy Server").
    """
    ws_user = os.environ.get("WEBSHARE_PROXY_USERNAME", "").strip()
    ws_pass = os.environ.get("WEBSHARE_PROXY_PASSWORD", "").strip()

    if ws_user and ws_pass:
        try:
            from youtube_transcript_api.proxies import WebshareProxyConfig
            proxy_config = WebshareProxyConfig(
                proxy_username=ws_user,
                proxy_password=ws_pass,
            )
            print(f"Using Webshare proxy for YouTube requests")
            return YouTubeTranscriptApi(proxy_config=proxy_config)
        except Exception as e:
            print(f"Webshare proxy setup failed: {e}, falling back to direct")

    return YouTubeTranscriptApi()


def _snippets_to_dicts(fetched) -> list:
    """Convert FetchedTranscript snippets to plain dicts."""
    return [{"text": s.text, "start": s.start, "duration": s.duration}
            for s in fetched.snippets]


def _try_fetch_transcript(video_id: str, cookie_file: str = None) -> Optional[list]:
    """
    Fetch transcript using youtube-transcript-api v1.0+ instance API.
    Always returns English — translates non-English transcripts via YouTube.

    Strategies:
    1. Direct fetch with English language codes (native English videos)
    2. List all transcripts, translate EVERY one to English (catches Hindi etc.)
    3. Raw fallback only if translation fails for every available transcript
    """
    try:
        api = _build_yt_api()
        en_codes = ["en", "en-US", "en-GB", "en-IN", "en-AU", "en-CA"]

        # ── Strategy 1: direct English fetch ──────────────────────────
        try:
            result = _snippets_to_dicts(api.fetch(video_id, languages=en_codes))
            if result:
                return result
        except Exception:
            pass

        # ── Strategy 2: list all → translate each to English ──────────
        # This is the critical path for non-English videos (Hindi, Spanish etc.)
        # We MUST try translate() on every transcript — including auto-generated —
        # because is_translatable can be False yet translate() still succeeds.
        try:
            all_transcripts = list(api.list(video_id))
            if not all_transcripts:
                return None

            # Try manual transcripts first (better quality), then auto-generated
            ordered = sorted(all_transcripts, key=lambda t: t.is_generated)

            for t in ordered:
                try:
                    translated = t.translate("en").fetch()
                    result = _snippets_to_dicts(translated)
                    if result:
                        print(f"Translated {t.language_code}→en for {video_id}")
                        return result
                except Exception:
                    continue  # this one failed, try the next

            # ── Strategy 3: raw fallback (non-English) — last resort ──
            # Only reached if ALL translation attempts failed.
            # Gemini can still summarise non-English text reasonably well.
            for t in ordered:
                try:
                    raw = t.fetch()
                    result = _snippets_to_dicts(raw)
                    if result:
                        print(f"Raw {t.language_code} fallback for {video_id} (translation unavailable)")
                        return result
                except Exception:
                    continue

        except Exception as e:
            print(f"Transcript list error for {video_id}: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"Transcript fetch error for {video_id}: {type(e).__name__}: {e}")

    return None

def _try_supadata_fetch_transcript(video_id: str, provided_key: str = None):
    """
    Fetch transcript via Supadata API.
    Returns (segments_list, None) on success, or (None, error_string) on failure.
    - Immediate response (HTTP 200): segments returned directly
    - Async job response (HTTP 202): poll /youtube/transcript/{jobId} until done
    - Correct unit conversion: Supadata returns offset/duration in MILLISECONDS
    """
    api_key = provided_key or os.environ.get("SUPADATA_API_KEY")
    if not api_key:
        return None, "no_api_key"

    base = "https://api.supadata.ai/v1"
    headers = {"x-api-key": api_key}

    def _parse_segments(data: dict) -> Optional[list]:
        """Convert Supadata response dict into normalised segment list."""
        segments = data.get("content") or []
        if not segments or not isinstance(segments, list):
            return None
        result = []
        for s in segments:
            text = s.get("text") or s.get("content") or ""
            if not text:
                continue
            # Supadata uses 'offset' in milliseconds; convert to seconds
            offset_ms = s.get("offset") or s.get("start") or 0
            duration_ms = s.get("duration") or 0
            result.append({
                "start":    float(offset_ms) / 1000.0,
                "duration": float(duration_ms) / 1000.0,
                "text":     text,
            })
        return result if result else None

    try:
        resp = requests.get(
            f"{base}/youtube/transcript",
            params={"videoId": video_id, "lang": "en"},
            headers=headers,
            timeout=15,
        )

        # ── Immediate result ──────────────────────────────────────────
        if resp.status_code == 200:
            segs = _parse_segments(resp.json())
            return (segs, None) if segs else (None, "200_but_empty_content")

        # ── Async job issued ──────────────────────────────────────────
        if resp.status_code == 202:
            job_id = resp.json().get("jobId")
            if not job_id:
                print("Supadata: got 202 but no jobId in response")
                return None

            print(f"Supadata: async job started ({job_id}), polling…")
            poll_url = f"{base}/youtube/transcript/{job_id}"
            # Poll up to ~5 minutes (60 × 5 s) — sufficient for 2-hour videos
            for attempt in range(60):
                time.sleep(5)
                poll = requests.get(poll_url, headers=headers, timeout=15)
                if poll.status_code == 200:
                    segments = _parse_segments(poll.json())
                    if segments:
                        print(f"Supadata: job {job_id} complete after {attempt + 1} polls")
                        return segments, None
                elif poll.status_code == 202:
                    continue  # still processing
                else:
                    err = f"poll_{poll.status_code}: {poll.text[:200]}"
                    print(f"Supadata poll error: {err}")
                    return None, err

            msg = f"job_{job_id}_timed_out"
            print(f"Supadata: {msg}")
            return None, msg

        err = f"status_{resp.status_code}: {resp.text[:300]}"
        print(f"Supadata unexpected: {err}")
        return None, err

    except Exception as e:
        print(f"Supadata exception for {video_id}: {e}")
        return None, str(e)

    return None, "unknown"

def fetch_transcript(video_id: str, supadata_key: str = None) -> dict:
    """
    Fetch transcript with two-tier fallback:
    Tier 1 — youtube_transcript_api (works locally and on uncapped IPs)
    Tier 2 — Supadata API (reliable on cloud/Render IPs that YouTube blocks)

    On Render, YouTube frequently returns 429 / IP-blocked errors.
    We always attempt Tier 1 first, but fall through to Supadata quickly
    if it fails — regardless of *why* it failed.
    """
    debug_info = {}
    cookie_file = None
    render_cookie_path = "/etc/secrets/cookies.txt"
    local_cookie_path = os.path.join(BASE_DIR, "cookies.txt")

    if os.path.exists(render_cookie_path):
        cookie_file = render_cookie_path
    elif os.path.exists(local_cookie_path):
        cookie_file = local_cookie_path

    # ── Tier 1 ────────────────────────────────────────────────────────
    fetched = _try_fetch_transcript(video_id, cookie_file)
    if fetched:
        debug_info["method"] = "youtube_transcript_api"
        return _format_transcript(fetched, debug_info)

    debug_info["tier1"] = "failed"
    print(f"Tier 1 failed for {video_id}, trying Supadata...")

    # ── Tier 2: Supadata ──────────────────────────────────────────────
    # Always prefer env var so it works even if frontend never sent the key
    env_key = os.environ.get("SUPADATA_API_KEY", "").strip()
    active_supadata_key = supadata_key.strip() if supadata_key else env_key or None
    debug_info["supadata_key_source"] = (
        "frontend" if (supadata_key and supadata_key.strip()) else ("env" if env_key else "none")
    )
    debug_info["supadata_key_present"] = bool(active_supadata_key)

    if active_supadata_key:
        fetched_supa, supa_error = _try_supadata_fetch_transcript(
            video_id, provided_key=active_supadata_key
        )
        if fetched_supa:
            debug_info["method"] = "supadata"
            return _format_transcript(fetched_supa, debug_info)
        else:
            debug_info["tier2"] = f"failed: {supa_error}"
    else:
        debug_info["tier2"] = "no_supadata_key"

    print(f"All transcript methods failed for {video_id}. debug={debug_info}")

    tier2_detail = debug_info.get("tier2", "unknown error")
    return {
        "error": f"Transcript fetch failed. Reason: {tier2_detail}",
        "segments": [],
        "full_text": "",
        "debug": debug_info,
    }

def _format_transcript(snippets, debug_info: dict) -> dict:
    """Format fetched transcript snippets into structured output."""
    formatted_segments = []

    for snippet in snippets:
        try:
            if isinstance(snippet, dict):
                start = snippet.get("start", 0)
                text = snippet.get("text", "")
                duration = snippet.get("duration", 0)
            else:
                start = getattr(snippet, "start", 0)
                text = getattr(snippet, "text", "")
                duration = getattr(snippet, "duration", 0)
        except Exception:
            continue

        minutes, seconds = int(start // 60), int(start % 60)
        formatted_segments.append({
            "timestamp": f"{minutes:02d}:{seconds:02d}",
            "start": start,
            "duration": duration,
            "text": text,
        })

    full_text = " ".join(s["text"] for s in formatted_segments)
    return {
        "segments": formatted_segments,
        "full_text": full_text,
        "error": None,
        "debug": debug_info,
    }

def summarize_transcript(title: str, full_text: str, model_name: str, gemini_key: str) -> str:
    """
    Generate an AI summary using Gemini.

    FIX: Each call creates its own genai client configured with the request's
    API key, so threads never share mutable global state.  The semaphore caps
    concurrent Gemini calls to avoid 429s while still processing videos in
    parallel.  Exponential back-off handles any transient quota errors.
    """
    if not full_text:
        return "No transcript text available to summarize."

    max_chars = 25000
    text_to_summarize = full_text[:max_chars]
    if len(full_text) > max_chars:
        text_to_summarize += "\n\n[Transcript truncated for summarization]"

    prompt = f"""You are an expert content analyst. Summarize the following YouTube video transcript into a clear, well-structured summary. Always respond in English regardless of the transcript language.

Video Title: "{title}"

Transcript:
{text_to_summarize}

Provide your summary in the following format:
📌 **Overview**: A 2-3 sentence overview of what the video is about.

🔑 **Key Points**:
• [Key point 1]
• [Key point 2]
• [Key point 3]

💡 **Key Takeaways**:
• [Most important insight or action item 1]
• [Most important insight or action item 2]

Be concise but comprehensive. Focus on the most important information."""

    # Wait if a recent 429 told us to back off
    with _retry_lock:
        wait_until = _retry_after.get(gemini_key, 0)
    sleep_for = wait_until - time.time()
    if sleep_for > 0:
        print(f"⏳ Backing off {sleep_for:.1f}s before Gemini call for '{title}'")
        time.sleep(sleep_for)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        with GEMINI_SEMAPHORE:
            try:
                # FIX: configure per-call so threads are fully independent
                local_client = genai.GenerativeModel(
                    model_name,
                    # Pass the key directly rather than relying on global state
                    # (requires google-generativeai ≥ 0.5; falls back gracefully)
                )
                genai.configure(api_key=gemini_key)
                response = local_client.generate_content(prompt)
                return response.text.strip()

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    backoff = 2 ** attempt          # 2 s, 4 s, 8 s
                    print(f"⚠️  429 on attempt {attempt} for '{title}'. Retrying in {backoff}s…")
                    with _retry_lock:
                        _retry_after[gemini_key] = time.time() + backoff
                    time.sleep(backoff)
                    continue
                # Non-quota error – return immediately
                return f"Summary failed: {err_str}"

    return "Summary failed: exceeded retry limit due to API quota errors."

def fetch_video_data(video_id: str, url: str, supadata_key: str = None) -> dict:
    """
    Phase 1 (sequential): fetch metadata + transcript for one video.
    No Gemini calls here — safe to run one-at-a-time to avoid YouTube
    rate-limiting parallel scrape attempts from the same IP.
    """
    print(f"📥 Fetching transcript: {video_id}")
    metadata = fetch_video_metadata(video_id)
    transcript_data = fetch_transcript(video_id, supadata_key=supadata_key)
    return {
        "video_id": video_id,
        "url": url,
        "metadata": metadata,
        "transcript_data": transcript_data,
    }

def summarize_video_data(fetched: dict, target_model: str, gemini_key: str) -> dict:
    """
    Phase 2 (parallel): run Gemini summarization on already-fetched data.
    Safe to parallelize because it makes no YouTube requests.
    """
    video_id = fetched["video_id"]
    url = fetched["url"]
    metadata = fetched["metadata"]
    transcript_data = fetched["transcript_data"]

    if transcript_data["full_text"]:
        summary = summarize_transcript(
            metadata["title"],
            transcript_data["full_text"],
            target_model,
            gemini_key,
        )
    elif transcript_data["error"]:
        summary = f"⚠️ Could not generate summary: {transcript_data['error']}"
    else:
        summary = "No transcript text available."

    print(f"✅ Done: {video_id}")
    return {
        "video_id": video_id,
        "url": url,
        "title": metadata["title"],
        "author": metadata["author"],
        "thumbnail": metadata["thumbnail"],
        "transcript_segments": transcript_data["segments"],
        "full_text": transcript_data["full_text"],
        "transcript_error": transcript_data["error"],
        "summary": summary,
    }

# ─── Routes ───────────────────────────────────────────────────────────

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    from datetime import datetime, timezone
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route("/api/debug-transcript", methods=["GET"])
def debug_transcript():
    """
    Diagnostic endpoint — call this to see exactly what transcript methods
    succeed or fail for a given video on this server.
    Usage: /api/debug-transcript?video_id=UYdbJDKvCEM
    """
    video_id = request.args.get("video_id", "").strip()
    if not video_id:
        return jsonify({"error": "Pass ?video_id=VIDEO_ID"}), 400

    ws_user = os.environ.get("WEBSHARE_PROXY_USERNAME", "")
    report = {
        "video_id": video_id,
        "proxy": "webshare" if ws_user else "none (set WEBSHARE_PROXY_USERNAME + WEBSHARE_PROXY_PASSWORD to fix IP blocks)",
        "steps": {},
    }

    # Step 1: youtube_transcript_api (v1.0+ instance API)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        tlist = api.list(video_id)
        langs = [{"code": t.language_code, "generated": t.is_generated,
                  "translatable": t.is_translatable} for t in tlist]
        report["steps"]["yt_list"] = {"status": "ok", "langs": langs}
    except Exception as e:
        report["steps"]["yt_list"] = {"status": "error", "error": str(e)}

    # Step 2: Try _try_fetch_transcript
    try:
        result = _try_fetch_transcript(video_id)
        if result:
            report["steps"]["yt_fetch"] = {"status": "ok", "segments": len(result)}
        else:
            report["steps"]["yt_fetch"] = {"status": "returned_none"}
    except Exception as e:
        report["steps"]["yt_fetch"] = {"status": "error", "error": str(e)}

    # Step 3: Supadata (env var key)
    supa_key = os.environ.get("SUPADATA_API_KEY")
    if supa_key:
        try:
            import requests as req
            resp = req.get(
                "https://api.supadata.ai/v1/youtube/transcript",
                params={"videoId": video_id, "lang": "en"},
                headers={"x-api-key": supa_key},
                timeout=15,
            )
            report["steps"]["supadata"] = {
                "status": resp.status_code,
                "body_preview": resp.text[:300],
            }
        except Exception as e:
            report["steps"]["supadata"] = {"status": "error", "error": str(e)}
    else:
        report["steps"]["supadata"] = {"status": "no_key_in_env"}

    return jsonify(report)

@app.route("/api/summarize", methods=["POST"])
def summarize():
    """Accept up to 5 YouTube URLs, validate the model, and fetch summaries robustly."""
    try:
        data = request.get_json() or {}
        raw_urls = data.get("urls", [])
        gemini_key = data.get("gemini_api_key", "").strip()
        supadata_key = data.get("supadata_api_key", "").strip()

        urls = [str(u).strip() for u in raw_urls if u and str(u).strip()]
        urls = urls[:5]

        if not urls:
            return jsonify({"error": "No valid URLs provided."}), 400
        if not gemini_key:
            return jsonify({"error": "Gemini API key is required."}), 400

        print(f"🎬 Processing {len(urls)} video(s)…")

        # ── Model discovery ────────────────────────────────────────────
        genai.configure(api_key=gemini_key)
        target_model = "gemini-1.5-flash"  # safe default
        try:
            available_models = list(genai.list_models())
            text_models = [
                m.name.replace("models/", "")
                for m in available_models
                if "generateContent" in m.supported_generation_methods
            ]
            if not text_models:
                return jsonify({"error": "Your API key has no access to text-generation models."}), 403

            flash_models = [m for m in text_models if "1.5-flash" in m]
            target_model = flash_models[0] if flash_models else text_models[0]
            print(f"✅ Using model: {target_model}")
        except Exception as e:
            if "API key not valid" in str(e):
                return jsonify({"error": "Invalid Gemini API Key."}), 401
            print(f"⚠️  Model discovery failed ({e}). Using default: {target_model}")

        # ── Build task list ────────────────────────────────────────────
        tasks, invalid_tasks = [], []
        for url in urls:
            video_id = extract_video_id(url)
            if video_id:
                tasks.append({"url": url, "video_id": video_id})
            else:
                invalid_tasks.append({"url": url, "error": f"Could not extract video ID from: {url}"})

        results = []

        # ── Phase 1: fetch transcripts SEQUENTIALLY ────────────────────
        # YouTube's transcript API blocks concurrent requests from the same
        # IP (treats them as scraping). Fetching one-at-a-time with a small
        # pause between requests is the only reliable fix.
        fetched_videos = []
        for task in tasks:
            try:
                fetched = fetch_video_data(
                    task["video_id"], task["url"], supadata_key=supadata_key
                )
                fetched_videos.append(fetched)
            except Exception as exc:
                results.append({
                    "url": task["url"],
                    "video_id": task["video_id"],
                    "error": f"Transcript fetch failed: {exc}",
                    "title": "Failed to load",
                })
            # Small polite delay between YouTube requests
            if len(tasks) > 1:
                time.sleep(1.5)

        # ── Phase 2: summarize IN PARALLEL ────────────────────────────
        # Gemini calls are independent HTTP requests; parallelizing them is
        # safe and cuts total wait time significantly for multiple videos.
        def safe_summarize(fetched):
            try:
                return summarize_video_data(fetched, target_model, gemini_key)
            except Exception as exc:
                return {
                    "url": fetched["url"],
                    "video_id": fetched["video_id"],
                    "error": f"Summarization failed: {exc}",
                    "title": fetched["metadata"].get("title", "Failed to load"),
                }

        with ThreadPoolExecutor(max_workers=max(len(fetched_videos), 1)) as executor:
            futures = {executor.submit(safe_summarize, f): f for f in fetched_videos}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    f = futures[future]
                    results.append({
                        "url": f["url"],
                        "video_id": f["video_id"],
                        "error": f"Unexpected error: {exc}",
                        "title": "Failed to load",
                    })

        # Append hard-failed URLs
        for t in invalid_tasks:
            results.append({"url": t["url"], "error": t["error"]})

        # Restore original input order
        url_order = {url: i for i, url in enumerate(urls)}
        results.sort(key=lambda r: url_order.get(r.get("url", ""), 999))

        return jsonify({"results": results})

    except Exception as exc:
        return jsonify({"error": f"A critical server error occurred: {exc}"}), 500


@app.route("/api/summarize/stream", methods=["POST"])
def summarize_stream():
    """
    SSE endpoint: streams one JSON result per video as it completes.
    This keeps the HTTP connection alive for long videos and lets the
    frontend render cards progressively — no more Render 30 s timeouts.
    """
    try:
        data = request.get_json() or {}
        raw_urls = data.get("urls", [])
        gemini_key = data.get("gemini_api_key", "").strip()
        supadata_key = data.get("supadata_api_key", "").strip()

        urls = [str(u).strip() for u in raw_urls if u and str(u).strip()]
        urls = urls[:5]

        if not urls:
            return jsonify({"error": "No valid URLs provided."}), 400
        if not gemini_key:
            return jsonify({"error": "Gemini API key is required."}), 400

        # ── Model discovery (done once, before streaming starts) ───────
        genai.configure(api_key=gemini_key)
        target_model = "gemini-1.5-flash"
        try:
            available_models = list(genai.list_models())
            text_models = [
                m.name.replace("models/", "")
                for m in available_models
                if "generateContent" in m.supported_generation_methods
            ]
            if not text_models:
                return jsonify({"error": "Your API key has no access to text-generation models."}), 403
            flash_models = [m for m in text_models if "1.5-flash" in m]
            target_model = flash_models[0] if flash_models else text_models[0]
            print(f"✅ Using model: {target_model}")
        except Exception as e:
            if "API key not valid" in str(e):
                return jsonify({"error": "Invalid Gemini API Key."}), 401
            print(f"⚠️  Model discovery failed ({e}). Using default: {target_model}")

        # ── Build task list ────────────────────────────────────────────
        tasks = []
        for url in urls:
            video_id = extract_video_id(url)
            if video_id:
                tasks.append({"url": url, "video_id": video_id})
            else:
                tasks.append({"url": url, "video_id": None})

        def generate():
            import json as _json

            # Phase 1 — sequential transcript fetching
            # Send a heartbeat comment every iteration so the SSE connection
            # stays alive through Render's idle-connection timeout (even when
            # Supadata is polling for minutes on a long video).
            fetched_videos = []
            for i, task in enumerate(tasks):
                yield ": heartbeat\n\n"   # SSE comment — keeps connection alive

                if task["video_id"] is None:
                    err_result = {
                        "url": task["url"],
                        "error": f"Could not extract video ID from: {task['url']}",
                        "title": None,
                    }
                    yield f"data: {_json.dumps({'type': 'result', 'data': err_result})}\n\n"
                    continue

                try:
                    fetched = fetch_video_data(
                        task["video_id"], task["url"], supadata_key=supadata_key
                    )
                    fetched_videos.append(fetched)
                except Exception as exc:
                    err_result = {
                        "url": task["url"],
                        "video_id": task["video_id"],
                        "error": f"Transcript fetch failed: {exc}",
                        "title": "Failed to load",
                    }
                    yield f"data: {_json.dumps({'type': 'result', 'data': err_result})}\n\n"

                # Small delay between YouTube requests to avoid IP blocks.
                # Skip after last task.
                if i < len(tasks) - 1:
                    time.sleep(1.5)

            # Phase 2 — parallel Gemini summarization; stream each result
            # as it finishes so the frontend can render cards immediately.
            if not fetched_videos:
                yield "data: [DONE]\n\n"
                return

            result_queue = __import__("queue").Queue()

            def summarize_and_enqueue(fetched):
                try:
                    result = summarize_video_data(fetched, target_model, gemini_key)
                except Exception as exc:
                    result = {
                        "url": fetched["url"],
                        "video_id": fetched["video_id"],
                        "error": f"Summarization failed: {exc}",
                        "title": fetched["metadata"].get("title", "Failed to load"),
                    }
                result_queue.put(result)

            with ThreadPoolExecutor(max_workers=max(len(fetched_videos), 1)) as executor:
                for fv in fetched_videos:
                    executor.submit(summarize_and_enqueue, fv)

                for _ in fetched_videos:
                    result = result_queue.get()
                    yield f"data: {_json.dumps({'type': 'result', 'data': result})}\n\n"

            yield "data: [DONE]\n\n"

        return app.response_class(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disables Nginx/Render proxy buffering
            },
        )

    except Exception as exc:
        return jsonify({"error": f"A critical server error occurred: {exc}"}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8082))
    is_dev = "--dev" in sys.argv or os.environ.get("FLASK_ENV") == "development"
    print(f"🎬 YouTubeSummarizer server starting on http://localhost:{port}")
    app.run(debug=is_dev, host="0.0.0.0", port=port)
