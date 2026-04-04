"""YouTubeSummarizer — Flask API server for YouTube transcript summarization."""

import os
import re
import sys
import json
import socket
import requests
import urllib3.util.connection as urllib3_cn
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Force IPv4 to bypass severe IPv6 DNS routing timeouts on local macOS DEV environments only.
# Render natively handles IPv6 seamlessly.
if not os.environ.get("RENDER"):
    def allowed_gai_family():
        return socket.AF_INET
    urllib3_cn.allowed_gai_family = allowed_gai_family

from youtube_transcript_api import YouTubeTranscriptApi

import google.generativeai as genai

# Secure deterministic file paths for remote static deployments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)


# ─── Helpers ──────────────────────────────────────────────────────────
from typing import Optional


def extract_video_id(url: str) -> Optional[str]:
    """Extract the YouTube video ID from various URL formats."""
    url = url.strip()
    patterns = [
        r"(?:v=|/v/)([\w-]{11})",                    # youtube.com/watch?v=ID
        r"youtu\.be/([\w-]{11})",                     # youtu.be/ID
        r"(?:embed/|shorts/)([\w-]{11})",             # embed or shorts
        r"^([\w-]{11})$",                             # bare ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_video_metadata(video_id: str) -> dict:
    """Fetch video title and thumbnail via oembed (no API key needed)."""
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
        }
        resp = requests.get(oembed_url, timeout=10, headers=headers)
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


def fetch_transcript(video_id: str) -> dict:
    """Fetch transcript for a YouTube video and ensure it is in English.
    
    Uses a multi-tier strategy:
    1. Direct English fetch with browser headers.
    2. List available transcripts and translate the best candidate (e.g. Hindi) to English.
    3. Fallback to any available transcript and rely on Gemini for translation.
    """
    debug_info = {}
    try:
        import http.cookiejar
        session = requests.Session()
        
        # Browser-like headers to bypass anti-bot measures on cloud IPs
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })

        cookie_file = os.path.join(BASE_DIR, "cookies.txt")
        debug_info["cookie_path"] = cookie_file
        debug_info["cookie_exists"] = os.path.exists(cookie_file)
        
        if debug_info["cookie_exists"]:
            try:
                cookie_jar = http.cookiejar.MozillaCookieJar(cookie_file)
                cookie_jar.load(ignore_discard=True, ignore_expires=True)
                session.cookies.update(cookie_jar)
                debug_info["cookies_loaded"] = len(cookie_jar)
            except Exception as ce:
                debug_info["cookie_load_error"] = str(ce)
        
        ytt_api = YouTubeTranscriptApi(http_client=session)
        fetched = None
        detected_language = None

        # ── Strategy 1: Try fetching English directly (preferred) ──
        try:
            fetched = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB", "en-IN"])
            detected_language = "English"
        except Exception:
            pass

        # ── Strategy 2: List and translate ──
        if fetched is None:
            try:
                transcript_list = ytt_api.list(video_id)
                
                # Check for English first in the list
                try:
                    t = transcript_list.find_transcript(["en", "en-US", "en-GB", "en-IN"])
                    fetched = t.fetch()
                    detected_language = "English"
                except:
                    # If no native English, try translating any available transcript to English
                    for t in transcript_list:
                        if t.is_translatable:
                            try:
                                fetched = t.translate("en").fetch()
                                detected_language = f"Translated to English (from {t.language_code})"
                                break
                            except:
                                continue
                
                # Final fallback: just get whichever transcript is available (even if not in English/translatable)
                if fetched is None:
                    for t in transcript_list:
                        try:
                            fetched = t.fetch()
                            detected_language = f"Original: {t.language_code}"
                            break
                        except:
                            continue

            except Exception as e:
                debug_info["list_error"] = str(e)

        if fetched is None:
            return {
                "error": "No transcript available for this video.", 
                "segments": [], 
                "full_text": "",
                "debug": debug_info
            }

        # Build timestamped segments from FetchedTranscriptSnippet dataclass objects
        formatted_segments = []
        for snippet in fetched:
            start = snippet.start
            text = snippet.text
            duration = getattr(snippet, 'duration', 0)

            minutes = int(start // 60)
            seconds = int(start % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            formatted_segments.append({
                "timestamp": timestamp,
                "start": start,
                "duration": duration,
                "text": text,
            })

        # Build full text
        full_text = " ".join(s["text"] for s in formatted_segments)

        return {
            "segments": formatted_segments,
            "full_text": full_text,
            "error": None,
        }

    except Exception as e:
        return {
            "error": f"Failed to fetch transcript: {str(e)}",
            "segments": [],
            "full_text": "",
        }


def get_best_model(gemini_key: str) -> str:
    """Auto-detect the best available Gemini model."""
    genai.configure(api_key=gemini_key)
    try:
        models = genai.list_models()
        flash_models = []
        for m in models:
            methods = m.supported_generation_methods
            if "generateContent" in methods and "flash" in m.name.lower():
                flash_models.append(m.name)

        if flash_models:
            flash_models.sort(reverse=True)
            best = flash_models[0]
            return best.replace("models/", "") if best.startswith("models/") else best
    except Exception as e:
        print(f"Could not list models: {e}")

    # Fallback list
    return "gemini-2.5-flash-preview-04-17"


# Cache the detected model name per API key
_model_cache = {}


def summarize_transcript(gemini_key: str, title: str, full_text: str) -> str:
    """Generate an AI summary of the transcript using Gemini."""
    if not gemini_key:
        return "No API key provided — cannot generate summary."

    if not full_text:
        return "No transcript text available to summarize."

    genai.configure(api_key=gemini_key)

    # Get or detect the best model
    if gemini_key not in _model_cache:
        _model_cache[gemini_key] = get_best_model(gemini_key)
    model_name = _model_cache[gemini_key]
    model = genai.GenerativeModel(model_name)

    # Truncate very long transcripts to avoid token limits
    max_chars = 30000
    text_to_summarize = full_text[:max_chars]
    if len(full_text) > max_chars:
        text_to_summarize += "\n\n[Transcript truncated for summarization]"

    prompt = f"""You are an expert content analyst. Summarize the following YouTube video transcript.
The transcript may be in English or another language (like Hindi), but you MUST always provide the summary in English.

Video Title: "{title}"

Transcript:
{text_to_summarize}

Provide your summary in the following format:
📌 **Overview**: A 2-3 sentence overview of what the video is about.

🔑 **Key Points**:
• [Key point 1]
• [Key point 2]
• [Key point 3]
• [Key point 4]
• [Key point 5]

💡 **Key Takeaways**:
• [Most important insight or action item 1]
• [Most important insight or action item 2]
• [Most important insight or action item 3]

Be concise but comprehensive. Focus on the most important information."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        _model_cache.pop(gemini_key, None)
        return f"Summary generation failed: {str(e)}"


def process_single_video(video_id: str, url: str, gemini_key: str) -> dict:
    """Process a single video: fetch metadata, transcript, and summary."""
    metadata = fetch_video_metadata(video_id)
    transcript_data = fetch_transcript(video_id)

    summary = ""
    if transcript_data["full_text"]:
        summary = summarize_transcript(gemini_key, metadata["title"], transcript_data["full_text"])
    elif transcript_data["error"]:
        summary = f"⚠️ Could not generate summary: {transcript_data['error']}"

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
    # Use app.static_folder to ensure a fully resolved absolute path is used.
    # If the file still 404s, it means the static/ folder is literally missing entirely on the remote deployment.
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    from datetime import datetime, timezone
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})


@app.route("/api/summarize", methods=["POST"])
def summarize():
    """Accept up to 5 YouTube URLs, fetch transcripts, and return summaries."""
    data = request.get_json() or {}
    urls = data.get("urls", [])
    gemini_key = data.get("gemini_api_key", "")

    if not urls:
        return jsonify({"error": "No URLs provided."}), 400

    if len(urls) > 5:
        return jsonify({"error": "Maximum 5 URLs allowed."}), 400

    if not gemini_key:
        return jsonify({"error": "Gemini API key is required."}), 400

    # Validate and extract video IDs
    tasks = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        video_id = extract_video_id(url)
        if not video_id:
            tasks.append({
                "url": url,
                "error": f"Could not extract video ID from: {url}",
            })
        else:
            tasks.append({
                "url": url,
                "video_id": video_id,
            })

    # Process valid videos concurrently
    results = []
    valid_tasks = [t for t in tasks if "video_id" in t]
    invalid_tasks = [t for t in tasks if "error" in t]

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {}
        for task in valid_tasks:
            future = executor.submit(
                process_single_video,
                task["video_id"],
                task["url"],
                gemini_key,
            )
            future_map[future] = task

        for future in as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "url": task["url"],
                    "video_id": task["video_id"],
                    "error": str(e),
                })

    # Add invalid URL errors
    for t in invalid_tasks:
        results.append({
            "url": t["url"],
            "error": t["error"],
        })

    # Sort results in the same order as input URLs
    url_order = {url.strip(): i for i, url in enumerate(urls)}
    results.sort(key=lambda r: url_order.get(r.get("url", ""), 999))

    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8082))
    is_dev = "--dev" in sys.argv or os.environ.get("FLASK_ENV") == "development"
    print(f"🎬 YouTubeSummarizer server starting on http://localhost:{port}")
    app.run(debug=is_dev, host="0.0.0.0", port=port)
