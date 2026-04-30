/**
 * Vikas.ai — Dashboard Application Logic
 * Connects to the FastAPI backend and drives all UI interactions.
 */

const API_BASE = "http://localhost:8000";

// ── Language metadata ──────────────────────────────────────
const LANGUAGES = [
  { code: "hi", name: "Hindi", native: "हिन्दी", flag: "🇮🇳" },
  { code: "bn", name: "Bengali", native: "বাংলা", flag: "🇮🇳" },
  { code: "ta", name: "Tamil", native: "தமிழ்", flag: "🇮🇳" },
  { code: "te", name: "Telugu", native: "తెలుగు", flag: "🇮🇳" },
  { code: "mr", name: "Marathi", native: "मराठी", flag: "🇮🇳" },
  { code: "gu", name: "Gujarati", native: "ગુજરાતી", flag: "🇮🇳" },
  { code: "kn", name: "Kannada", native: "ಕನ್ನಡ", flag: "🇮🇳" },
  { code: "ml", name: "Malayalam", native: "മലയാളം", flag: "🇮🇳" },
  { code: "pa", name: "Punjabi", native: "ਪੰਜਾਬੀ", flag: "🇮🇳" },
  { code: "or", name: "Odia", native: "ଓଡ଼ିଆ", flag: "🇮🇳" },
  { code: "as", name: "Assamese", native: "অসমীয়া", flag: "🇮🇳" },
  { code: "ur", name: "Urdu", native: "اردو", flag: "🇮🇳" },
  { code: "en", name: "English", native: "English", flag: "🌐" },
];

// Seed documents for the knowledge base table
const SEED_DOCS = [
  { id: "nih-headache-001", source: "NIH — Neurological Disorders", domain: "medical", preview: "Tension-type headaches are the most common form…" },
  { id: "who-fever-002", source: "WHO — Clinical Guidelines", domain: "medical", preview: "Fever is defined as body temperature above 38°C…" },
  { id: "nimhans-mental-003", source: "NIMHANS — India", domain: "mental_health", preview: "Anxiety disorders are characterised by excessive worry…" },
  { id: "govt-disability-004", source: "Dept. of PwD, Govt. of India", domain: "civic", preview: "Under the RPwD Act 2016, individuals with 40%+ disability…" },
  { id: "first-aid-burn-005", source: "Red Cross — First Aid", domain: "medical", preview: "For minor burns: cool under running water for 10 min…" },
];

// ── DOM References ─────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Tab Navigation ─────────────────────────────────────────
$$(".nav-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".nav-tab").forEach((t) => t.classList.remove("active"));
    $$(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`#tab-${tab.dataset.tab}`).classList.add("active");
  });
});

// ── Health Check ───────────────────────────────────────────
let healthData = null;

async function checkHealth() {
  const dot = $("#server-status-dot");
  const text = $("#server-status-text");
  try {
    const resp = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
    healthData = await resp.json();
    dot.className = "status-dot online";
    text.textContent = `v${healthData.version}`;
    updateKPIs(healthData);
  } catch {
    dot.className = "status-dot offline";
    text.textContent = "Offline";
    updateKPIsOffline();
  }
}

function updateKPIs(data) {
  const docsEl = $("#kpi-docs-value");
  const groqEl = $("#kpi-groq-value");
  const vapiEl = $("#kpi-vapi-value");
  const twilioEl = $("#kpi-twilio-value");
  if (docsEl) docsEl.textContent = data.knowledge_base_docs ?? "—";
  if (groqEl) { groqEl.textContent = data.groq_configured ? "✓ Ready" : "✗ Missing"; groqEl.style.color = data.groq_configured ? "var(--accent-emerald)" : "var(--accent-rose)"; }
  if (vapiEl) { vapiEl.textContent = data.vapi_configured ? "✓ Ready" : "✗ Missing"; vapiEl.style.color = data.vapi_configured ? "var(--accent-emerald)" : "var(--accent-rose)"; }
  if (twilioEl) { twilioEl.textContent = data.twilio_configured ? "✓ Ready" : "✗ Missing"; twilioEl.style.color = data.twilio_configured ? "var(--accent-emerald)" : "var(--accent-rose)"; }
}

function updateKPIsOffline() {
  ["#kpi-docs-value", "#kpi-groq-value", "#kpi-vapi-value", "#kpi-twilio-value"].forEach((sel) => {
    const el = $(sel);
    if (el) { el.textContent = "—"; el.style.color = ""; }
  });
}

// ── Language Grid ──────────────────────────────────────────
function renderLanguageGrid() {
  const grid = $("#lang-grid");
  if (!grid) return;
  grid.innerHTML = LANGUAGES.map(
    (l) => `<div class="lang-chip"><span class="lang-flag">${l.flag}</span><span>${l.native} <span style="color:var(--text-muted);font-size:0.72rem">(${l.name})</span></span></div>`
  ).join("");
}

// ── Knowledge Base Table ───────────────────────────────────
function renderKBTable() {
  const tbody = $("#kb-table-body");
  tbody.innerHTML = SEED_DOCS.map(
    (d) => `<tr><td>${d.id}</td><td>${d.source}</td><td><span class="domain-badge">${d.domain}</span></td><td>${d.preview}</td></tr>`
  ).join("");
}

// ── Ingest Form Toggle ─────────────────────────────────────
$("#kb-ingest-btn").addEventListener("click", () => {
  const form = $("#ingest-form");
  form.style.display = form.style.display === "none" ? "block" : "none";
});

// ── Ingest Submit ──────────────────────────────────────────
$("#ingest-submit").addEventListener("click", async () => {
  const source = $("#ingest-source").value.trim();
  const domain = $("#ingest-domain").value;
  const content = $("#ingest-content").value.trim();
  if (!source || !content) return alert("Please fill in source and content.");

  try {
    const resp = await fetch(`${API_BASE}/api/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        documents: [{
          id: `custom-${Date.now()}`,
          content,
          source,
          metadata: { domain },
        }],
      }),
    });
    const data = await resp.json();
    alert(`✅ Ingested ${data.documents_ingested} document(s).`);
    $("#ingest-source").value = "";
    $("#ingest-content").value = "";
    checkHealth();
  } catch (err) {
    alert(`❌ Ingestion failed: ${err.message}`);
  }
});

// ── Example Chips ──────────────────────────────────────────
$$(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("#query-message").value = chip.dataset.msg;
    // Switch to query tab
    $$(".nav-tab").forEach((t) => t.classList.remove("active"));
    $$(".tab-panel").forEach((p) => p.classList.remove("active"));
    $("#nav-query").classList.add("active");
    $("#tab-query").classList.add("active");
  });
});

// ── Pipeline Inspector State ───────────────────────────────
function resetPipeline() {
  ["pipe-intake", "pipe-retrieval", "pipe-reasoning", "pipe-synthesis", "pipe-guardrails"].forEach((id) => {
    const node = $(`#${id}`);
    node.classList.remove("active", "done", "skipped");
    const output = $(`#${id}-output`);
    if (output) { output.classList.remove("visible"); output.textContent = ""; }
  });
  $("#pipe-edge-1-label").textContent = "";
}

function setPipelineNode(id, state, outputText) {
  const node = $(`#${id}`);
  node.classList.remove("active", "done", "skipped");
  if (state) node.classList.add(state);
  if (outputText) {
    const output = $(`#${id}-output`);
    if (output) { output.textContent = outputText; output.classList.add("visible"); }
  }
}

// ── Query Submit ───────────────────────────────────────────
$("#query-submit").addEventListener("click", async () => {
  const message = $("#query-message").value.trim();
  const language = $("#query-language").value;
  if (!message) return;

  const btn = $("#query-submit");
  const btnText = btn.querySelector(".btn-text");
  const btnLoader = btn.querySelector(".btn-loader");
  btn.disabled = true;
  btnText.style.display = "none";
  btnLoader.style.display = "inline-flex";

  // Reset and activate pipeline
  resetPipeline();
  setPipelineNode("pipe-intake", "active");

  // Switch pipeline tab to show activity
  $("#result-placeholder").style.display = "none";
  $("#result-content").style.display = "none";

  try {
    // Simulate pipeline stages with timed updates
    const stageTimers = [
      setTimeout(() => { setPipelineNode("pipe-intake", "done", "Triage complete"); setPipelineNode("pipe-retrieval", "active"); }, 800),
      setTimeout(() => { setPipelineNode("pipe-retrieval", "done", "Documents retrieved"); setPipelineNode("pipe-reasoning", "active"); }, 2000),
      setTimeout(() => { setPipelineNode("pipe-reasoning", "done", "Reasoning complete"); setPipelineNode("pipe-synthesis", "active"); }, 3500),
    ];

    const resp = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, language }),
    });

    const data = await resp.json();

    // Clear simulated timers if they haven't fired
    stageTimers.forEach(clearTimeout);

    // Mark all pipeline nodes
    const isEmergency = data.severity === "emergency";
    setPipelineNode("pipe-intake", "done", `Domain: ${data.domain} | Severity: ${data.severity}`);
    if (isEmergency) {
      $("#pipe-edge-1-label").textContent = "EMERGENCY → skip to synthesis";
      setPipelineNode("pipe-retrieval", "skipped");
      setPipelineNode("pipe-reasoning", "skipped");
    } else {
      $("#pipe-edge-1-label").textContent = "normal flow";
      setPipelineNode("pipe-retrieval", "done", `${data.citations.length} source(s) retrieved`);
      setPipelineNode("pipe-reasoning", "done", `${data.reasoning_steps.length} reasoning step(s) | Confidence: ${(data.confidence * 100).toFixed(0)}%`);
    }
    setPipelineNode("pipe-synthesis", "done", "Response generated");
    setPipelineNode("pipe-guardrails", "done", data.disclaimer_injected ? "⚠️ Disclaimer injected" : "✓ Output passed validation");

    // Render result
    renderResult(data);
  } catch (err) {
    $("#result-placeholder").style.display = "flex";
    alert(`❌ Query failed: ${err.message}\n\nMake sure the backend is running on ${API_BASE}`);
  } finally {
    btn.disabled = false;
    btnText.style.display = "inline";
    btnLoader.style.display = "none";
  }
});

function renderResult(data) {
  $("#result-placeholder").style.display = "none";
  $("#result-content").style.display = "block";

  // Severity
  const sevBadge = $("#result-severity");
  sevBadge.textContent = data.severity.toUpperCase();
  sevBadge.className = `severity-badge ${data.severity}`;

  // Domain
  $("#result-domain").textContent = data.domain.replace("_", " ");

  // Confidence
  const confPct = Math.round(data.confidence * 100);
  $("#result-confidence-bar").style.width = `${confPct}%`;
  $("#result-confidence-value").textContent = `${confPct}%`;

  // Response text
  $("#result-response-text").textContent = data.response;

  // Disclaimer
  $("#result-disclaimer").style.display = data.disclaimer_injected ? "block" : "none";

  // Reasoning steps
  const stepsEl = $("#result-reasoning-steps");
  stepsEl.innerHTML = data.reasoning_steps.length
    ? data.reasoning_steps.map((s) => `<li>${escapeHtml(s)}</li>`).join("")
    : "<li>No reasoning steps recorded (emergency fast-track).</li>";

  // Citations
  const citEl = $("#result-citations");
  citEl.innerHTML = data.citations.length
    ? data.citations.map((c) => `<li>📄 ${escapeHtml(c)}</li>`).join("")
    : "<li>No citations (emergency response uses hard-coded safety data).</li>";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ── Init ───────────────────────────────────────────────────
renderLanguageGrid();
renderKBTable();
checkHealth();
setInterval(checkHealth, 15000);

// ── Recordings — OTP Auth + Call History ────────────────────

let verifiedPhone = null;

$("#rec-send-otp").addEventListener("click", async () => {
  const phone = $("#rec-phone").value.trim();
  if (!phone) return;

  const status = $("#rec-auth-status");
  status.textContent = "Sending OTP…";
  status.style.color = "var(--text-secondary)";

  try {
    const resp = await fetch(`${API_BASE}/api/auth/request-otp`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phone_number: phone }),
    });
    const data = await resp.json();
    if (data.status === "ok") {
      status.textContent = "✓ OTP sent! Check your phone.";
      status.style.color = "var(--accent-emerald)";
      $("#otp-input-section").style.display = "block";
    } else {
      status.textContent = "✗ Failed to send OTP";
      status.style.color = "var(--accent-rose)";
    }
  } catch (err) {
    status.textContent = `✗ Error: ${err.message}`;
    status.style.color = "var(--accent-rose)";
  }
});

$("#rec-verify-otp").addEventListener("click", async () => {
  const phone = $("#rec-phone").value.trim();
  const code = $("#rec-otp").value.trim();
  if (!phone || !code) return;

  const status = $("#rec-auth-status");
  status.textContent = "Verifying…";

  try {
    const resp = await fetch(`${API_BASE}/api/auth/verify-otp`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phone_number: phone, code }),
    });
    const data = await resp.json();

    if (data.status === "ok") {
      verifiedPhone = phone;
      $("#recordings-auth").style.display = "none";
      $("#recordings-list").style.display = "block";
      $("#rec-phone-display").textContent = `Recordings for ${phone}`;
      renderRecordings(data.recordings || []);
    } else {
      status.textContent = "✗ " + (data.message || "Invalid OTP");
      status.style.color = "var(--accent-rose)";
    }
  } catch (err) {
    status.textContent = `✗ Error: ${err.message}`;
    status.style.color = "var(--accent-rose)";
  }
});

$("#rec-logout").addEventListener("click", () => {
  verifiedPhone = null;
  $("#recordings-auth").style.display = "block";
  $("#recordings-list").style.display = "none";
  $("#rec-phone").value = "";
  $("#rec-otp").value = "";
  $("#otp-input-section").style.display = "none";
  $("#rec-auth-status").textContent = "";
});

function renderRecordings(recordings) {
  const container = $("#rec-cards");
  const emptyMsg = $("#rec-empty");

  if (!recordings.length) {
    container.innerHTML = "";
    emptyMsg.style.display = "block";
    return;
  }

  emptyMsg.style.display = "none";
  container.innerHTML = recordings
    .map(
      (r) => `
    <div class="glass-card" style="margin-bottom: 1rem;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
        <strong style="font-size: 0.9rem;">${escapeHtml(r.id || "Call")}</strong>
        <span style="font-size: 0.8rem; color: var(--text-secondary);">${r.created_at || ""}</span>
      </div>
      ${r.summary ? `<p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.75rem;">${escapeHtml(r.summary)}</p>` : ""}
      ${r.duration ? `<p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">Duration: ${r.duration}s</p>` : ""}
      ${
        r.recording
          ? `<audio controls preload="none" style="width: 100%; margin-bottom: 0.5rem;"><source src="${escapeHtml(r.recording)}" type="audio/wav">Your browser does not support audio.</audio>`
          : `<p style="font-size: 0.85rem; color: var(--text-muted);">No recording available</p>`
      }
      ${
        r.transcript
          ? `<details style="margin-top: 0.5rem;"><summary style="cursor: pointer; font-size: 0.85rem; color: var(--text-secondary);">View Transcript</summary><pre style="white-space: pre-wrap; font-size: 0.85rem; margin-top: 0.5rem; padding: 0.75rem; background: var(--bg-hover); border-radius: var(--radius-sm);">${escapeHtml(r.transcript)}</pre></details>`
          : ""
      }
    </div>
  `
    )
    .join("");
}
