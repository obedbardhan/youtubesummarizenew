/**
 * YouTubeSummarizer — Frontend Logic
 * Handles URL validation, API calls, and dynamic result rendering
 */

// ─── State ──────────────────────────────────────────────────────────
const API_KEY_STORAGE_KEY = "yt_summarizer_gemini_key";

// ─── DOM References ─────────────────────────────────────────────────
const settingsBtn = document.getElementById("settingsBtn");
const settingsModal = document.getElementById("settingsModal");
const closeSettings = document.getElementById("closeSettings");
const apiKeyInput = document.getElementById("apiKeyInput");
const toggleKeyVisibility = document.getElementById("toggleKeyVisibility");
const saveKeyBtn = document.getElementById("saveKeyBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const urlCount = document.getElementById("urlCount");
const loadingSection = document.getElementById("loadingSection");
const loadingText = document.getElementById("loadingText");
const resultsSection = document.getElementById("resultsSection");
const resultsContainer = document.getElementById("resultsContainer");
const newBatchBtn = document.getElementById("newBatchBtn");

// ─── Initialization ─────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initSettings();
    initUrlInputs();
    initActions();
});

// ─── Settings ───────────────────────────────────────────────────────
function initSettings() {
    // Load saved key
    const savedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
    if (savedKey) {
        apiKeyInput.value = savedKey;
    }

    // Open modal
    settingsBtn.addEventListener("click", () => {
        settingsModal.classList.add("active");
        apiKeyInput.focus();
    });

    // Close modal
    closeSettings.addEventListener("click", () => {
        settingsModal.classList.remove("active");
    });

    settingsModal.addEventListener("click", (e) => {
        if (e.target === settingsModal) {
            settingsModal.classList.remove("active");
        }
    });

    // Toggle visibility
    toggleKeyVisibility.addEventListener("click", () => {
        const isPassword = apiKeyInput.type === "password";
        apiKeyInput.type = isPassword ? "text" : "password";
    });

    // Save key
    saveKeyBtn.addEventListener("click", () => {
        const key = apiKeyInput.value.trim();
        if (key) {
            localStorage.setItem(API_KEY_STORAGE_KEY, key);
            settingsModal.classList.remove("active");
            showToast("API key saved successfully!", "success");
        } else {
            showToast("Please enter a valid API key.", "error");
        }
    });

    // Enter key to save
    apiKeyInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") saveKeyBtn.click();
    });
}

// ─── URL Input Handling ─────────────────────────────────────────────
function initUrlInputs() {
    const urlInputs = document.querySelectorAll(".url-input");
    const clearBtns = document.querySelectorAll(".clear-url-btn");

    urlInputs.forEach((input) => {
        input.addEventListener("input", () => {
            updateUrlState(input);
            updateUrlCount();
        });

        input.addEventListener("paste", () => {
            // Delay to let paste complete
            setTimeout(() => {
                updateUrlState(input);
                updateUrlCount();
            }, 50);
        });

        input.addEventListener("blur", () => {
            validateUrlInput(input);
        });
    });

    clearBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const idx = btn.dataset.index;
            const input = document.querySelector(`.url-input[data-index="${idx}"]`);
            input.value = "";
            updateUrlState(input);
            updateUrlCount();
            input.classList.remove("valid", "invalid");
            input.focus();
        });
    });
}

function updateUrlState(input) {
    const row = input.closest(".url-input-row");
    if (input.value.trim()) {
        row.classList.add("has-value");
    } else {
        row.classList.remove("has-value");
        input.classList.remove("valid", "invalid");
    }
}

function validateUrlInput(input) {
    const value = input.value.trim();
    if (!value) {
        input.classList.remove("valid", "invalid");
        return;
    }

    if (isValidYouTubeUrl(value)) {
        input.classList.add("valid");
        input.classList.remove("invalid");
    } else {
        input.classList.add("invalid");
        input.classList.remove("valid");
    }
}

function isValidYouTubeUrl(url) {
    const patterns = [
        /(?:youtube\.com\/watch\?v=)([\w-]{11})/,
        /(?:youtu\.be\/)([\w-]{11})/,
        /(?:youtube\.com\/embed\/)([\w-]{11})/,
        /(?:youtube\.com\/shorts\/)([\w-]{11})/,
        /^[\w-]{11}$/,
    ];
    return patterns.some((p) => p.test(url));
}

function getValidUrls() {
    const inputs = document.querySelectorAll(".url-input");
    const urls = [];
    inputs.forEach((input) => {
        const value = input.value.trim();
        if (value && isValidYouTubeUrl(value)) {
            urls.push(value);
        }
    });
    return urls;
}

function updateUrlCount() {
    const validUrls = getValidUrls();
    const allFilled = document.querySelectorAll(".url-input-row.has-value").length;
    urlCount.textContent = `${allFilled} / 5 URLs entered`;

    summarizeBtn.disabled = validUrls.length === 0;
}

// ─── Actions ────────────────────────────────────────────────────────
function initActions() {
    summarizeBtn.addEventListener("click", handleSummarize);
    newBatchBtn.addEventListener("click", handleNewBatch);
    
    // Download All button is in the HTML, hook it up here
    const downloadAllBtn = document.getElementById("downloadAllBtn");
    if (downloadAllBtn) {
        downloadAllBtn.addEventListener("click", handleDownloadAll);
    }
}

async function handleSummarize() {
    const geminiKey = localStorage.getItem(API_KEY_STORAGE_KEY);
    if (!geminiKey) {
        showToast("Please set your Gemini API key in Settings first.", "error");
        settingsModal.classList.add("active");
        return;
    }

    const urls = getValidUrls();
    if (urls.length === 0) {
        showToast("Please enter at least one valid YouTube URL.", "error");
        return;
    }

    // Show loading
    document.querySelector(".input-section").style.display = "none";
    resultsSection.style.display = "none";
    loadingSection.style.display = "block";
    loadingText.textContent = `Fetching transcripts and generating summaries for ${urls.length} video${urls.length > 1 ? "s" : ""}...`;

    try {
        const response = await fetch("/api/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                urls: urls,
                gemini_api_key: geminiKey,
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Server error");
        }

        renderResults(data.results);

    } catch (error) {
        showToast(`Error: ${error.message}`, "error");
        document.querySelector(".input-section").style.display = "block";
    } finally {
        loadingSection.style.display = "none";
    }
}

function handleNewBatch() {
    resultsSection.style.display = "none";
    document.querySelector(".input-section").style.display = "block";

    // Clear all inputs
    document.querySelectorAll(".url-input").forEach((input) => {
        input.value = "";
        input.classList.remove("valid", "invalid");
        updateUrlState(input);
    });
    updateUrlCount();
    resultsContainer.innerHTML = "";
}

// ─── Render Results ─────────────────────────────────────────────────
function renderResults(results) {
    resultsContainer.innerHTML = "";

    results.forEach((result, index) => {
        if (result.error && !result.title) {
            // Error card
            resultsContainer.appendChild(createErrorCard(result));
        } else {
            // Video card
            resultsContainer.appendChild(createVideoCard(result, index));
        }
    });

    // Make results global so Download All can access them easily
    window.currentResults = results;

    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function createErrorCard(result) {
    const card = document.createElement("div");
    card.className = "error-card";
    
    let debugHtml = "";
    if (result.debug) {
        const d = result.debug;
        debugHtml = `
            <div class="debug-info" style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px; font-family: monospace; font-size: 11px; opacity: 0.8; text-align: left; border-left: 3px solid #ff4b2b;">
                <div style="margin-bottom: 5px; color: #ff4b2b; font-weight: bold;">[Diagnostic Info]</div>
                • Path: ${escapeHtml(d.cookie_path || "unknown")}<br>
                • Exists: ${d.cookie_exists ? "✅ Yes" : "❌ No"}<br>
                • Cookies Loaded: ${d.cookies_loaded || 0}<br>
                ${d.cookie_load_error ? `• Error: ${escapeHtml(d.cookie_load_error)}` : ""}
            </div>
        `;
    }

    card.innerHTML = `
        <div class="error-title">❌ Failed to process video</div>
        <div class="error-message">${escapeHtml(result.error)}</div>
        ${debugHtml}
        <div class="error-url">${escapeHtml(result.url || "")}</div>
    `;
    return card;
}

function createVideoCard(result, index) {
    const card = document.createElement("div");
    card.className = "video-card";
    card.dataset.index = index;

    const videoUrl = `https://www.youtube.com/watch?v=${result.video_id}`;

    // Header
    const header = document.createElement("div");
    header.className = "video-card-header";
    header.innerHTML = `
        <img class="video-thumbnail" src="${escapeHtml(result.thumbnail)}" alt="Video thumbnail" 
             onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22120%22 height=%2268%22%3E%3Crect fill=%22%231a1a2e%22 width=%22120%22 height=%2268%22/%3E%3Ctext x=%2260%22 y=%2238%22 fill=%22%235a5e72%22 text-anchor=%22middle%22 font-size=%2212%22%3ENo thumb%3C/text%3E%3C/svg%3E'">
        <div class="video-info">
            <div class="video-title"><a href="${videoUrl}" target="_blank" rel="noopener">${escapeHtml(result.title)}</a></div>
            <div class="video-author">${escapeHtml(result.author)}</div>
        </div>
    `;

    // Body — split panel
    const body = document.createElement("div");
    body.className = "video-card-body";

    // Summary Panel
    const summaryPanel = document.createElement("div");
    summaryPanel.className = "summary-panel";
    summaryPanel.innerHTML = `
        <div class="panel-header">
            <div class="panel-title"><span class="icon">📝</span> Summary</div>
            <button class="copy-btn" data-copy="summary" onclick="copyText(this, 'summary')">📋 Copy</button>
        </div>
        <div class="summary-content">
            ${formatSummary(result.summary || "No summary available.")}
            ${result.debug && (result.summary && result.summary.includes("⚠️")) ? `
                <div class="debug-info" style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 6px; font-family: monospace; font-size: 11px; opacity: 0.7;">
                    <strong>[Cookie Diagnostic]</strong><br>
                    • File: ${result.debug.cookie_exists ? "✅ Found" : "❌ Not Found"}<br>
                    • Path: ${escapeHtml(result.debug.cookie_path)}<br>
                    • Count: ${result.debug.cookies_loaded || 0}<br>
                    ${result.debug.cookie_load_error ? `<span style="color: #ff4b2b;">• Load Error: ${escapeHtml(result.debug.cookie_load_error)}</span>` : ""}
                </div>
            ` : ""}
        </div>
    `;

    // Transcript Panel
    const transcriptPanel = document.createElement("div");
    transcriptPanel.className = "transcript-panel";

    if (result.transcript_segments && result.transcript_segments.length > 0) {
        let segmentsHtml = result.transcript_segments.map((seg) => `
            <div class="transcript-segment">
                <span class="segment-time" title="Jump to ${seg.timestamp}" onclick="window.open('${videoUrl}&t=${Math.floor(seg.start)}', '_blank')">${seg.timestamp}</span>
                <span class="segment-text">${escapeHtml(seg.text)}</span>
            </div>
        `).join("");

        transcriptPanel.innerHTML = `
            <div class="panel-header">
                <div class="panel-title"><span class="icon">📋</span> Full Transcript</div>
                <button class="copy-btn" onclick="copyText(this, 'transcript')">📋 Copy</button>
            </div>
            <div class="transcript-scroll">${segmentsHtml}</div>
        `;

        // Store full text for copy
        transcriptPanel.dataset.fullText = result.full_text || "";
    } else {
        transcriptPanel.innerHTML = `
            <div class="panel-header">
                <div class="panel-title"><span class="icon">📋</span> Full Transcript</div>
            </div>
            <div class="no-transcript">
                <div class="icon">🚫</div>
                <p>${result.transcript_error ? escapeHtml(result.transcript_error) : "No transcript available for this video."}</p>
                ${result.debug ? `
                    <div class="debug-info" style="margin-top: 10px; padding: 8px; background: rgba(0,0,0,0.1); border-radius: 4px; font-family: monospace; font-size: 10px; text-align: left; border-left: 2px solid #5a5e72;">
                        [Debug] Path: ${escapeHtml(result.debug.cookie_path)} | Exists: ${result.debug.cookie_exists} | Count: ${result.debug.cookies_loaded}
                    </div>
                ` : ""}
            </div>
        `;
    }

    body.appendChild(summaryPanel);
    body.appendChild(transcriptPanel);

    // Download footer (per-card)
    const downloadFooter = document.createElement("div");
    downloadFooter.className = "download-footer";
    downloadFooter.innerHTML = `
        <div class="download-options">
            <label class="download-checkbox">
                <input type="checkbox" checked class="include-summary" data-index="${index}">
                <span class="checkbox-label">Summary</span>
            </label>
            <label class="download-checkbox">
                <input type="checkbox" checked class="include-transcript" data-index="${index}">
                <span class="checkbox-label">Transcript</span>
            </label>
        </div>
        <button class="download-card-btn" onclick="handleCardDownload(${index})">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Download
        </button>
    `;

    card.appendChild(header);
    card.appendChild(body);
    card.appendChild(downloadFooter);

    return card;
}

// ─── Copy functionality ─────────────────────────────────────────────
function copyText(btn, type) {
    const card = btn.closest(".video-card");
    let text = "";

    if (type === "summary") {
        const summaryEl = card.querySelector(".summary-content");
        text = summaryEl.innerText;
    } else if (type === "transcript") {
        const transcriptPanel = card.querySelector(".transcript-panel");
        text = transcriptPanel.dataset.fullText || transcriptPanel.innerText;
    }

    navigator.clipboard.writeText(text).then(() => {
        btn.classList.add("copied");
        const originalText = btn.innerHTML;
        btn.innerHTML = "✅ Copied!";
        setTimeout(() => {
            btn.classList.remove("copied");
            btn.innerHTML = originalText;
        }, 2000);
    }).catch(() => {
        showToast("Failed to copy to clipboard", "error");
    });
}

// ─── Download File Generation ───────────────────────────────────────
function downloadTextFile(filename, text) {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
}

function getCardTextComponent(result, includeSummary, includeTranscript) {
    let content = [];
    content.push(`🎬 Title: ${result.title || "Unknown"}`);
    content.push(`🔗 URL: ${result.url || ""}`);
    content.push(`👤 Author: ${result.author || "Unknown"}`);
    content.push(`--------------------------------------------------\n`);

    if (includeSummary && result.summary) {
        content.push(`✨ AI SUMMARY\n==================================================\n`);
        content.push(`${result.summary}\n\n`);
    }

    if (includeTranscript && result.full_text) {
        content.push(`📝 FULL TRANSCRIPT\n==================================================\n`);
        content.push(`${result.full_text}\n\n`);
    }

    return content.join("\n");
}

function handleCardDownload(index) {
    if (!window.currentResults || !window.currentResults[index]) return;
    const result = window.currentResults[index];
    
    // Check which options are selected for this card
    const card = document.querySelector(`.video-card[data-index="${index}"]`);
    const includeSummary = card.querySelector(".include-summary").checked;
    const includeTranscript = card.querySelector(".include-transcript").checked;

    if (!includeSummary && !includeTranscript) {
        showToast("Please select at least one item to download (Summary or Transcript).", "error");
        return;
    }

    const textContent = getCardTextComponent(result, includeSummary, includeTranscript);
    
    // Clean up filename
    let safeTitle = (result.title || "Video").replace(/[^a-z0-9]/gi, '_').substring(0, 30);
    let parts = [];
    if (includeSummary) parts.push("Summary");
    if (includeTranscript) parts.push("Transcript");
    
    const filename = `${safeTitle}_${parts.join('_')}.txt`;
    downloadTextFile(filename, textContent);
    showToast(`Downloaded ${parts.join(" & ")}`, "success");
}

function handleDownloadAll() {
    if (!window.currentResults || window.currentResults.length === 0) return;

    let totalContent = [`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`,
                        `         YOUTUBE SUMMARIZER BATCH EXPORT          `,
                        `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`];
    
    let processedAny = false;

    window.currentResults.forEach((result, idx) => {
        if (result.error && !result.title) return; // Skip complete failures
        totalContent.push(`\n[VIDEO ${idx + 1}]`);
        // Always include both for Download All
        totalContent.push(getCardTextComponent(result, true, true));
        totalContent.push(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);
        processedAny = true;
    });

    if (!processedAny) {
        showToast("Nothing to download.", "error");
        return;
    }

    const dateStr = new Date().toISOString().split('T')[0];
    const filename = `YouTubeSummarizer_AllResults_${dateStr}.txt`;
    
    downloadTextFile(filename, totalContent.join("\n"));
    showToast("Downloaded all results", "success");
}

// ─── Helpers ────────────────────────────────────────────────────────
function formatSummary(text) {
    // Convert markdown-like formatting to HTML
    let html = escapeHtml(text.toString());

    // Bold text **text**
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Bullet points
    html = html.replace(/^[•●*-]\s/gm, "• ");

    return html;
}

function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message, type = "info") {
    // Remove existing toasts
    document.querySelectorAll(".toast").forEach((t) => t.remove());

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add("show");
    });

    // Auto-remove
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 400);
    }, 3500);
}
