"""YouTubeSummarizer — Flask API server for YouTube transcript summarization."""

import os
import re
import sys
import socket
import requests
import urllib3.util.connection as urllib3_cn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai

# For processing multiple videos simultaneously
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

# ─── Helpers ──────────────────────────────────────────────────────────
from typing import Optional

def extract_video_id(url: str) -> Optional[str]:
    """Extract the YouTube video ID from various URL formats."""
    url = url.strip()
    patterns = [
        r"(?:v=|/v/)([\w-]{11})",
        r"youtu\.be/([\w-]{11})",
        r"(?:embed/|shorts/)([\w-]{11})",
        r"^([\w-]{11})$",
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

def _try_fetch_transcript(video_id: str, cookie_file: str = None) -> Optional[list]:
    """Attempt to fetch transcript, strictly forcing English natively or via translation."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookie_file)
        
        # 1. Try to find a native English transcript (manual or auto-generated)
        try:
            en_transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'en-IN'])
            return en_transcript.fetch()
        except Exception:
            pass # Move to translation phase
        
        # 2. If no English track exists, find ANY track and translate it to English
        for transcript in transcript_list:
            if transcript.is_translatable:
                translated = transcript.translate('en')
                return translated.fetch()
                
    except Exception as e:
        print(f"Transcript fetch/translate error: {e}")
    
    return None

def _try_supadata_fetch_transcript(video_id: str, provided_key: str = None) -> Optional[list]:
    """Fallback: Fetch transcript using Supadata API."""
    api_key = provided_key or os.environ.get("SUPADATA_API_KEY")
    if not api_key:
        return None

    try:
        url = f"https://api.supadata.ai/v1/youtube/transcript?videoId={video_id}&lang=en"
        headers = {"x-api-key": api_key}
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            segments = data.get("content") or data.get("transcript") or []
            if not segments and isinstance(data, list):
                segments = data
            
            if not segments:
                return None
            
            return [{
                "start": float(s.get("start") if s.get("start") is not None else s.get("offset", 0)), 
                "duration": float(s.get("duration") if s.get("duration") is not None else s.get("duration", 0)), 
                "text": s.get("text") or s.get("content", "")
            } for s in segments if s.get("text") or s.get("content")]
    except Exception as e:
        print(f"Supadata fetch failed: {e}")
    return None

def fetch_transcript(video_id: str, supadata_key: str = None) -> dict:
    """Fetch transcript prioritizing native translation."""
    debug_info = {}
    cookie_file = None
    render_cookie_path = "/etc/secrets/cookies.txt"
    local_cookie_path = os.path.join(BASE_DIR, "cookies.txt")
    
    if os.path.exists(render_cookie_path):
        cookie_file = render_cookie_path
    elif os.path.exists(local_cookie_path):
        cookie_file = local_cookie_path

    # ── Tier 1: YouTubeTranscriptApi (Best for Forced Translation) ──
    fetched = _try_fetch_transcript(video_id, cookie_file)
    if fetched:
        debug_info["method"] = "youtube_transcript_api_translated"
        return _format_transcript(fetched, debug_info)

    # ── Tier 2: Supadata (Fallback if blocked) ──
    try:
        active_supadata_key = supadata_key or os.environ.get("SUPADATA_API_KEY")
        if active_supadata_key:
            fetched_supa = _try_supadata_fetch_transcript(video_id, provided_key=active_supadata_key)
            if fetched_supa:
                debug_info["method"] = "supadata"
                return _format_transcript(fetched_supa, debug_info)
    except Exception as e:
        debug_info["tier2_error"] = str(e)

    # All tiers failed
    return {
        "error": "Could not extract or translate transcript. Video might lack captions.",
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
                start, text, duration = snippet.get("start", 0), snippet.get("text", ""), snippet.get("duration", 0)
            else:
                start, text, duration = getattr(snippet, "start", 0), getattr(snippet, "text", ""), getattr(snippet, "duration", 0)
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

_working_model_cache = {}

def summarize_transcript(gemini_key: str, title: str, full_text: str) -> str:
    """Generate an AI summary of the transcript using Gemini with Smart Auto-Discovery."""
    if not gemini_key:
        return "No API key provided — cannot generate summary."
    if not full_text:
        return "No transcript text available to summarize."

    try:
        genai.configure(api_key=gemini_key)

        max_chars = 25000 
        text_to_summarize = full_text[:max_chars]
        if len(full_text) > max_chars:
            text_to_summarize += "\n\n[Transcript truncated for summarization]"

        prompt = f"""You are an expert content analyst. Summarize the following YouTube video transcript into a clear, well-structured summary.

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

        # 1. Check if we already found a working model for this key
        if gemini_key in _working_model_cache:
            try:
                model = genai.GenerativeModel(_working_model_cache[gemini_key])
                return model.generate_content(prompt).text.strip()
            except Exception:
                # If the cached model suddenly fails, clear it and fall through to discovery
                del _working_model_cache[gemini_key]

        # 2. Try the standard 1.5 names first (fastest path)
        known_models = ["gemini-1.5-flash", "gemini-1.5-pro", "models/gemini-1.5-flash"]
        for model_name in known_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                _working_model_cache[gemini_key] = model_name # Save the winner
                return response.text.strip()
            except Exception as e:
                # If it's a 404, ignore and try the next one. If it's a real error (like quota), raise it.
                if "404" in str(e) or "not found" in str(e).lower():
                    continue
                raise e 

        # 3. Smart Auto-Discovery: Ask the API for available models if the standard names failed
        print("⚠️ Standard model names failed. Querying API for available models...")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                model_name = m.name.replace("models/", "") # Clean the prefix
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    _working_model_cache[gemini_key] = model_name # Save the winner
                    print(f"✅ Successfully auto-discovered and used model: {model_name}")
                    return response.text.strip()
                except Exception:
                    continue # Try the next one in the list

        return "Summary generation failed: Your API key does not have access to any text-generation models."

    except Exception as e:
        return f"Summary generation failed: {str(e)}"

def process_single_video(video_id: str, url: str, gemini_key: str, supadata_key: str = None) -> dict:
    """Process a single video: fetch metadata, transcript, and summary."""
    metadata = fetch_video_metadata(video_id)
    transcript_data = fetch_transcript(video_id, supadata_key=supadata_key)

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
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    from datetime import datetime, timezone
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route("/api/summarize", methods=["POST"])
def summarize():
    """Accept up to 5 YouTube URLs, fetch transcripts, and return summaries."""
    try:
        data = request.get_json() or {}
        urls = data.get("urls", [])
        gemini_key = data.get("gemini_api_key", "")
        supadata_key = data.get("supadata_api_key", "")

        if not urls: return jsonify({"error": "No URLs provided."}), 400
        if len(urls) > 5: return jsonify({"error": "Maximum 5 URLs allowed."}), 400
        if not gemini_key: return jsonify({"error": "Gemini API key is required."}), 400

        tasks = []
        for url in urls:
            url = url.strip()
            if not url: continue
            video_id = extract_video_id(url)
            if not video_id:
                tasks.append({"url": url, "error": f"Could not extract video ID from: {url}"})
            else:
                tasks.append({"url": url, "video_id": video_id})

        results = []
        valid_tasks = [t for t in tasks if "video_id" in t]
        invalid_tasks = [t for t in tasks if "error" in t]

        # 🚀 Process all valid videos concurrently using parallel threads
        def process_task(task):
            try:
                return process_single_video(task["video_id"], task["url"], gemini_key, supadata_key=supadata_key)
            except Exception as e:
                return {"url": task["url"], "video_id": task["video_id"], "error": f"Process error: {str(e)}", "title": "Failed to load"}

        # Run up to 5 tasks at the exact same time
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_task = {executor.submit(process_task, task): task for task in valid_tasks}
            for future in as_completed(future_to_task):
                results.append(future.result())

        for t in invalid_tasks:
            results.append({"url": t["url"], "error": t["error"]})

        # Restore the original order the user entered them in
        url_order = {url.strip(): i for i, url in enumerate(urls)}
        results.sort(key=lambda r: url_order.get(r.get("url", ""), 999))

        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": f"A server-side error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8082))
    is_dev = "--dev" in sys.argv or os.environ.get("FLASK_ENV") == "development"
    print(f"🎬 YouTubeSummarizer server starting on http://localhost:{port}")
    app.run(debug=is_dev, host="0.0.0.0", port=port)
