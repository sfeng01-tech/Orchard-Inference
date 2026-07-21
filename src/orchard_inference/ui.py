"""Dependency-free local UI for exercising Orchard inference."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

from orchard_inference.chunked_prefill import (
    ChunkedPrefillPolicy,
    ChunkedPrefillSimulator,
    deterministic_workload,
    summarize,
)
from orchard_inference.kv_blocks import deterministic_scenario

router = APIRouter()


class ChunkedPrefillBody(BaseModel):
    """UI request for the chunked-prefill simulator."""

    requests: int = Field(default=8, ge=1, le=128)
    arrival_interval_steps: int = Field(default=2, ge=0, le=128)
    prompt_tokens: int = Field(default=256, ge=1, le=8192)
    output_tokens: int = Field(default=32, ge=1, le=2048)
    prefix_saved_tokens: int = Field(default=128, ge=0, le=8192)
    chunk_size: int = Field(default=64, ge=1, le=2048)
    decode_tpot_slo_steps: int = Field(default=2, ge=1, le=128)


class KVBlocksBody(BaseModel):
    """UI request for the KV block manager simulator."""

    block_size_tokens: int = Field(default=16, ge=1, le=1024)
    sequences: int = Field(default=6, ge=1, le=128)
    base_prompt_tokens: int = Field(default=128, ge=0, le=8192)
    shared_prefix_tokens: int = Field(default=96, ge=0, le=8192)
    decode_tokens: int = Field(default=16, ge=0, le=2048)


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Orchard Control Room</title>
  <link rel="stylesheet" href="/ui/styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Orchard Control Room</h1>
      <p>Send prompts, stream tokens, and inspect serving metadata.</p>
    </div>
    <div class="status-strip">
      <span id="readyBadge" class="badge">checking</span>
      <span id="modelBadge" class="badge muted">model</span>
    </div>
  </header>

  <main class="layout">
    <section class="panel request-panel">
      <div class="panel-head">
        <h2>Request Console</h2>
        <button id="refreshBtn" type="button">Refresh</button>
      </div>
      <label>Model
        <select id="modelSelect"></select>
      </label>
      <label>System
        <textarea id="systemPrompt" rows="3">You are concise and technically precise.</textarea>
      </label>
      <label>User
        <textarea id="userPrompt" rows="7">Explain why bounded queues matter in an LLM
inference server.</textarea>
      </label>
      <div class="controls">
        <label>Max tokens
          <input id="maxTokens" type="number" min="1" max="4096" value="128">
        </label>
        <label>Temperature
          <input id="temperature" type="number" min="0" max="2" step="0.1" value="0">
        </label>
        <label class="checkline">
          <input id="streamToggle" type="checkbox" checked>
          Stream
        </label>
      </div>
      <button id="sendBtn" class="primary" type="button">Send request</button>
    </section>

    <section class="panel output-panel">
      <div class="panel-head">
        <h2>Token Stream</h2>
        <span id="requestState" class="badge muted">idle</span>
      </div>
      <pre id="tokenOutput" class="token-output"></pre>
      <div class="metrics-grid">
        <div><span>TTFT</span><strong id="ttft">-</strong></div>
        <div><span>Tokens</span><strong id="tokenCount">0</strong></div>
        <div><span>Elapsed</span><strong id="elapsed">-</strong></div>
        <div><span>Finish</span><strong id="finishReason">-</strong></div>
      </div>
    </section>

    <section class="panel">
      <h2>Lifecycle Approximation</h2>
      <ol id="timeline" class="timeline">
        <li data-stage="received">received</li>
        <li data-stage="validated">validated</li>
        <li data-stage="queued">queued</li>
        <li data-stage="scheduled">scheduled</li>
        <li data-stage="prefill">prefill</li>
        <li data-stage="decoding">decoding</li>
        <li data-stage="completed">completed</li>
      </ol>
    </section>

    <section class="panel">
      <h2>Prefix Router</h2>
      <dl class="kv">
        <div><dt>Route</dt><dd id="prefixRoute">-</dd></div>
        <div><dt>Matched tokens</dt><dd id="prefixMatched">-</dd></div>
        <div><dt>Matched ratio</dt><dd id="prefixRatio">-</dd></div>
        <div><dt>Estimated saved</dt><dd id="prefixSaved">-</dd></div>
      </dl>
    </section>

    <section class="panel">
      <h2>Scheduler Metadata</h2>
      <dl class="kv">
        <div><dt>Queue seconds</dt><dd id="queueSeconds">-</dd></div>
        <div><dt>Batch ID</dt><dd id="batchId">-</dd></div>
        <div><dt>Batch size</dt><dd id="batchSize">-</dd></div>
        <div><dt>Request ID</dt><dd id="requestId">-</dd></div>
      </dl>
    </section>

    <section class="panel observability-panel">
      <div class="panel-head">
        <h2>Observability Dashboard</h2>
        <span id="metricsUpdated" class="badge muted">metrics</span>
      </div>
      <div class="metrics-grid observability-cards">
        <div><span>Completed</span><strong id="metricCompleted">0</strong></div>
        <div><span>Failed</span><strong id="metricFailed">0</strong></div>
        <div><span>Timed out</span><strong id="metricTimedOut">0</strong></div>
        <div><span>Output tokens</span><strong id="metricOutputTokens">0</strong></div>
        <div><span>Queue depth</span><strong id="metricQueueDepth">0</strong></div>
        <div><span>Active</span><strong id="metricActive">0</strong></div>
        <div><span>Current batch</span><strong id="metricBatch">0</strong></div>
        <div><span>Prefix saved</span><strong id="metricPrefixSaved">0</strong></div>
      </div>
      <div class="chart-grid">
        <div>
          <h3>Request Outcomes</h3>
          <div id="requestBars" class="bars"></div>
        </div>
        <div>
          <h3>System Gauges</h3>
          <div id="gaugeBars" class="bars"></div>
        </div>
        <div>
          <h3>Prefix Router</h3>
          <div id="prefixBars" class="bars"></div>
        </div>
      </div>
      <h3>Selected Raw Metrics</h3>
      <div id="metricsSnapshot" class="metrics-list"></div>
    </section>

    <section class="panel simulator-panel">
      <div class="panel-head">
        <h2>Chunked Prefill Simulator</h2>
        <button id="runChunkedBtn" type="button">Run</button>
      </div>
      <div class="sim-controls">
        <label>Requests <input id="simRequests" type="number" min="1" value="8"></label>
        <label>Arrival gap <input id="simArrival" type="number" min="0" value="2"></label>
        <label>Prompt tokens <input id="simPrompt" type="number" min="1" value="256"></label>
        <label>Output tokens <input id="simOutput" type="number" min="1" value="32"></label>
        <label>Prefix saved <input id="simSaved" type="number" min="0" value="128"></label>
        <label>Chunk size <input id="simChunk" type="number" min="1" value="64"></label>
      </div>
      <div id="chunkedSummary" class="metrics-list"></div>
      <div id="chunkedChart" class="policy-chart"></div>
      <div id="chunkedTimeline" class="step-timeline"></div>
    </section>

    <section class="panel simulator-panel">
      <div class="panel-head">
        <h2>KV Block Manager</h2>
        <button id="runKvBtn" type="button">Run</button>
      </div>
      <div class="sim-controls">
        <label>Block size <input id="kvBlockSize" type="number" min="1" value="16"></label>
        <label>Sequences <input id="kvSequences" type="number" min="1" value="6"></label>
        <label>Base prompt <input id="kvPrompt" type="number" min="0" value="128"></label>
        <label>Shared prefix <input id="kvShared" type="number" min="0" value="96"></label>
        <label>Decode tokens <input id="kvDecode" type="number" min="0" value="16"></label>
      </div>
      <div id="kvSummary" class="metrics-list"></div>
      <div id="kvBlocks" class="kv-block-grid"></div>
      <div id="kvAllocations" class="metrics-list"></div>
    </section>
  </main>

  <script src="/ui/app.js"></script>
</body>
</html>
"""

CSS = """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #18202a;
  --muted: #697386;
  --line: #d9dee8;
  --accent: #0f766e;
  --accent-ink: #ffffff;
  --warn: #b45309;
  --bad: #b91c1c;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
}

.topbar {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: center;
  padding: 22px 28px;
  border-bottom: 1px solid var(--line);
  background: #ffffff;
}

h1, h2, p { margin: 0; }
h1 { font-size: 24px; }
h2 { font-size: 16px; }
p { color: var(--muted); margin-top: 4px; }

.layout {
  display: grid;
  grid-template-columns: minmax(320px, 420px) minmax(360px, 1fr);
  gap: 16px;
  padding: 16px;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}

.request-panel { grid-row: span 4; }
.output-panel { min-height: 380px; }
.observability-panel { grid-column: 2; }
.simulator-panel { grid-column: span 2; }

.panel-head, .status-strip, .controls {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
}

label {
  display: grid;
  gap: 6px;
  margin-top: 14px;
  color: var(--muted);
  font-size: 13px;
}

textarea, select, input {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  color: var(--ink);
  background: #fff;
  font: inherit;
}

.controls label { flex: 1; }
.checkline {
  display: flex;
  grid-template-columns: none;
  align-items: center;
  justify-content: center;
  margin-top: 30px;
}
.checkline input { width: auto; }

button {
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  padding: 9px 12px;
  cursor: pointer;
  font-weight: 600;
}

button.primary {
  width: 100%;
  margin-top: 16px;
  border-color: var(--accent);
  background: var(--accent);
  color: var(--accent-ink);
}

.badge {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 700;
  color: var(--accent);
}
.badge.muted { color: var(--muted); }
.badge.bad { color: var(--bad); }

.token-output {
  min-height: 240px;
  white-space: pre-wrap;
  line-height: 1.55;
  padding: 14px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #fbfcfe;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}
.metrics-grid div, .metrics-list div, .kv div {
  border-top: 1px solid var(--line);
  padding-top: 10px;
}
.metrics-grid span, .kv dt {
  display: block;
  color: var(--muted);
  font-size: 12px;
}
.metrics-grid strong, .kv dd {
  margin: 3px 0 0;
  font-size: 15px;
  overflow-wrap: anywhere;
}

.timeline {
  list-style: none;
  padding: 0;
  margin: 12px 0 0;
  display: grid;
  gap: 8px;
}
.timeline li {
  border-left: 4px solid var(--line);
  padding: 8px 10px;
  background: #fbfcfe;
  border-radius: 4px;
}
.timeline li.active { border-left-color: var(--accent); font-weight: 700; }
.timeline li.warn { border-left-color: var(--warn); }

.kv {
  display: grid;
  gap: 10px;
  margin: 12px 0 0;
}

.metrics-list {
  display: grid;
  gap: 8px;
  margin-top: 12px;
  font-size: 13px;
}

h3 {
  margin: 16px 0 8px;
  font-size: 13px;
  color: var(--muted);
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}

.bars {
  display: grid;
  gap: 8px;
}

.bar-row {
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr) 56px;
  gap: 8px;
  align-items: center;
  font-size: 12px;
}

.bar-track {
  height: 10px;
  background: #eef2f6;
  border-radius: 999px;
  overflow: hidden;
}

.bar-fill {
  display: block;
  height: 100%;
  width: 0%;
  background: var(--accent);
}

.sim-controls {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
}

.policy-chart {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.policy-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fbfcfe;
}

.policy-card h3 { margin-top: 0; color: var(--ink); }

.step-timeline {
  display: flex;
  gap: 3px;
  overflow-x: auto;
  margin-top: 14px;
  padding-bottom: 6px;
}

.step-cell {
  flex: 0 0 10px;
  height: 28px;
  border-radius: 3px;
  background: #e5e7eb;
}

.step-cell.prefill { background: #0f766e; }
.step-cell.decode { background: #2563eb; }
.step-cell.idle { background: #d1d5db; }

.kv-block-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(46px, 1fr));
  gap: 6px;
  margin-top: 14px;
}

.kv-block {
  border: 1px solid var(--line);
  border-radius: 6px;
  min-height: 42px;
  padding: 6px;
  background: #fbfcfe;
  font-size: 11px;
}

.kv-block.shared {
  border-color: var(--accent);
  background: #ecfdf5;
}

@media (max-width: 900px) {
  .topbar { align-items: flex-start; flex-direction: column; }
  .layout { grid-template-columns: 1fr; }
  .request-panel { grid-row: auto; }
  .observability-panel { grid-column: auto; }
  .simulator-panel { grid-column: auto; }
  .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .chart-grid { grid-template-columns: 1fr; }
  .sim-controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""

JS = """
const state = {
  startedAt: 0,
  firstTokenAt: 0,
  tokenCount: 0,
  timer: null,
};

const $ = (id) => document.getElementById(id);

function setBadge(id, text, kind = "muted") {
  const node = $(id);
  node.textContent = text;
  node.className = `badge ${kind}`;
}

function setStage(stage, warn = false) {
  for (const item of document.querySelectorAll("#timeline li")) {
    item.classList.toggle("active", item.dataset.stage === stage);
    item.classList.toggle("warn", warn && item.dataset.stage === stage);
  }
  $("requestState").textContent = stage;
}

function seconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Number(value).toFixed(4)}s`;
}

async function refreshStatus() {
  try {
    const ready = await fetch("/health/ready");
    setBadge("readyBadge", ready.ok ? "ready" : "not ready", ready.ok ? "" : "bad");
  } catch {
    setBadge("readyBadge", "offline", "bad");
  }
  try {
    const models = await fetch("/v1/models").then((r) => r.json());
    const select = $("modelSelect");
    select.innerHTML = "";
    for (const model of models.data || []) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${model.id} (${model.backend})`;
      select.appendChild(option);
    }
    const first = models.data?.[0];
    setBadge(
      "modelBadge",
      first ? `${first.backend} / ${first.quantization || "none"}` : "no model"
    );
  } catch {
    setBadge("modelBadge", "model unavailable", "bad");
  }
  await refreshMetrics();
}

async function refreshMetrics() {
  try {
    const text = await fetch("/metrics").then((r) => r.text());
    const metrics = parseMetrics(text);
    renderObservability(metrics);
    const wanted = [
      "orchard_requests_completed_total",
      "orchard_requests_failed_total",
      "orchard_requests_timed_out_total",
      "orchard_requests_rejected_total",
      "orchard_output_tokens_total",
      "orchard_queue_depth",
      "orchard_active_requests",
      "orchard_prefix_router_matched_tokens_total",
      "orchard_prefix_router_estimated_saved_prefill_tokens_total",
    ];
    const rows = text.split("\\n")
      .filter((line) => wanted.some((name) => line.startsWith(name)))
      .slice(0, 16);
    $("metricsSnapshot").innerHTML = rows.length
      ? rows.map((line) => `<div>${line}</div>`).join("")
      : "<div>No matching metrics yet.</div>";
  } catch {
    $("metricsSnapshot").innerHTML = "<div>Metrics unavailable.</div>";
  }
}

function parseMetrics(text) {
  const metrics = {};
  for (const line of text.split("\\n")) {
    if (!line || line.startsWith("#")) continue;
    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\\{[^}]*\\})?\\s+(.+)$/);
    if (!match) continue;
    const name = match[1];
    const labels = match[2] || "";
    const value = Number(match[3]);
    if (!Number.isFinite(value)) continue;
    metrics[`${name}${labels}`] = value;
    if (!labels) metrics[name] = value;
  }
  return metrics;
}

function metric(metrics, name) {
  return metrics[name] ?? 0;
}

function formatCount(value) {
  return Number(value || 0).toLocaleString(undefined, {maximumFractionDigits: 2});
}

function renderObservability(metrics) {
  $("metricCompleted").textContent = formatCount(
    metric(metrics, "orchard_requests_completed_total")
  );
  $("metricFailed").textContent = formatCount(metric(metrics, "orchard_requests_failed_total"));
  $("metricTimedOut").textContent = formatCount(
    metric(metrics, "orchard_requests_timed_out_total")
  );
  $("metricOutputTokens").textContent = formatCount(metric(metrics, "orchard_output_tokens_total"));
  $("metricQueueDepth").textContent = formatCount(metric(metrics, "orchard_queue_depth"));
  $("metricActive").textContent = formatCount(metric(metrics, "orchard_active_requests"));
  $("metricBatch").textContent = formatCount(metric(metrics, "orchard_current_batch_size"));
  $("metricPrefixSaved").textContent = formatCount(
    metric(metrics, "orchard_prefix_router_estimated_saved_prefill_tokens_total")
  );
  $("metricsUpdated").textContent = new Date().toLocaleTimeString();
  renderBars("requestBars", [
    ["completed", metric(metrics, "orchard_requests_completed_total")],
    ["failed", metric(metrics, "orchard_requests_failed_total")],
    ["timed out", metric(metrics, "orchard_requests_timed_out_total")],
    ["cancelled", metric(metrics, "orchard_requests_cancelled_total")],
  ]);
  renderBars("gaugeBars", [
    ["queue", metric(metrics, "orchard_queue_depth")],
    ["active", metric(metrics, "orchard_active_requests")],
    ["batch", metric(metrics, "orchard_current_batch_size")],
    ["loaded", metric(metrics, "orchard_loaded_models")],
  ]);
  renderBars("prefixBars", [
    ["matched", metric(metrics, "orchard_prefix_router_matched_tokens_total")],
    ["saved", metric(metrics, "orchard_prefix_router_estimated_saved_prefill_tokens_total")],
    ["hits", metric(metrics, 'orchard_prefix_router_requests_total{route="prefix_hit"}')],
    ["misses", metric(metrics, 'orchard_prefix_router_requests_total{route="prefix_miss"}')],
  ]);
}

function renderBars(id, rows) {
  const max = Math.max(1, ...rows.map(([, value]) => value));
  $(id).innerHTML = rows.map(([label, value]) => {
    const width = Math.max(2, Math.round((value / max) * 100));
    return `<div class="bar-row">
      <span>${label}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${width}%"></span></span>
      <strong>${formatCount(value)}</strong>
    </div>`;
  }).join("");
}

function numberValue(id) {
  const value = Number($(id).value);
  return Number.isFinite(value) ? value : 0;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(`Unexpected non-JSON response from ${url}: ${text.slice(0, 160)}`);
    }
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string"
      ? data.detail
      : JSON.stringify(data.detail || data);
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }
  return data;
}

async function runChunkedPrefill() {
  const button = $("runChunkedBtn");
  button.disabled = true;
  $("chunkedSummary").innerHTML = "<div>running chunked-prefill simulation…</div>";
  $("chunkedChart").innerHTML = "";
  $("chunkedTimeline").innerHTML = "";
  try {
    const body = {
      requests: numberValue("simRequests"),
      arrival_interval_steps: numberValue("simArrival"),
      prompt_tokens: numberValue("simPrompt"),
      output_tokens: numberValue("simOutput"),
      prefix_saved_tokens: numberValue("simSaved"),
      chunk_size: numberValue("simChunk"),
      decode_tpot_slo_steps: 2,
    };
    const data = await postJson("/ui/simulate/chunked-prefill", body);
    renderChunked(data);
  } catch (error) {
    $("chunkedSummary").innerHTML = `<div>chunked-prefill failed: ${String(error)}</div>`;
  } finally {
    button.disabled = false;
  }
}

function renderChunked(data) {
  if (!Array.isArray(data.runs) || data.runs.length === 0) {
    throw new Error("simulator returned no chunked-prefill runs");
  }
  $("chunkedSummary").innerHTML = data.runs.map((run) => {
    const summary = run.summary;
    return `<div>${run.policy}: TTFT p95 ${formatCount(summary.ttft_p95_steps)} steps,
      TPOT p95 ${formatCount(summary.tpot_p95_steps)} steps,
      violation ${formatCount(summary.tpot_violation_rate * 100)}%</div>`;
  }).join("");
  $("chunkedChart").innerHTML = data.runs.map((run) => {
    const rows = [
      ["TTFT p95", run.summary.ttft_p95_steps || 0],
      ["TPOT p95", run.summary.tpot_p95_steps || 0],
      ["latency p95", run.summary.latency_p95_steps || 0],
    ];
    const max = Math.max(1, ...rows.map(([, value]) => value));
    const bars = rows.map(([label, value]) => {
      const width = Math.max(2, Math.round((value / max) * 100));
      return `<div class="bar-row"><span>${label}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${width}%"></span></span>
        <strong>${formatCount(value)}</strong></div>`;
    }).join("");
    return `<div class="policy-card"><h3>${run.policy}</h3>${bars}</div>`;
  }).join("");
  const mixed = data.runs.find((run) => run.policy === "mixed_slo") || data.runs[0];
  $("chunkedTimeline").innerHTML = (mixed.events || []).map((event) => {
    const title = `step ${event.step}: ${event.operation} ${event.request_id || ""}`;
    return `<span class="step-cell ${event.operation}" title="${title}"></span>`;
  }).join("");
}

async function runKvBlocks() {
  const button = $("runKvBtn");
  button.disabled = true;
  $("kvSummary").innerHTML = "<div>running KV block simulation…</div>";
  $("kvBlocks").innerHTML = "";
  $("kvAllocations").innerHTML = "";
  try {
    const body = {
      block_size_tokens: numberValue("kvBlockSize"),
      sequences: numberValue("kvSequences"),
      base_prompt_tokens: numberValue("kvPrompt"),
      shared_prefix_tokens: numberValue("kvShared"),
      decode_tokens: numberValue("kvDecode"),
    };
    const data = await postJson("/ui/simulate/kv-blocks", body);
    renderKvBlocks(data);
  } catch (error) {
    $("kvSummary").innerHTML = `<div>KV block simulation failed: ${String(error)}</div>`;
  } finally {
    button.disabled = false;
  }
}

function renderKvBlocks(data) {
  const summary = data.summary;
  if (!summary || !Array.isArray(data.allocations)) {
    throw new Error("simulator returned invalid KV block data");
  }
  $("kvSummary").innerHTML = [
    ["physical blocks", summary.physical_blocks],
    ["capacity tokens", summary.physical_capacity_tokens],
    ["used tokens", summary.physical_used_tokens],
    ["dense capacity", summary.dense_capacity_tokens],
    ["saved vs dense", summary.estimated_capacity_tokens_saved_vs_dense],
    ["fragmentation", summary.internal_fragmentation_tokens],
  ].map(([label, value]) => `<div>${label}: ${formatCount(value)}</div>`).join("");
  const refcounts = {};
  for (const allocation of data.allocations) {
    for (const block of allocation.block_ids) refcounts[block] = (refcounts[block] || 0) + 1;
  }
  const blockIds = Object.keys(refcounts).map(Number).sort((a, b) => a - b);
  $("kvBlocks").innerHTML = blockIds.map((block) => {
    const shared = refcounts[block] > 1 ? "shared" : "";
    return `<div class="kv-block ${shared}">b${block}<br>ref ${refcounts[block]}</div>`;
  }).join("");
  $("kvAllocations").innerHTML = data.allocations.map((allocation) => {
    return `<div>${allocation.request_id}: [${allocation.block_ids.join(", ")}],
      shared ${allocation.shared_prefix_tokens},
      copied ${allocation.copied_prefix_tokens}</div>`;
  }).join("");
}

function resetRun() {
  state.startedAt = performance.now();
  state.firstTokenAt = 0;
  state.tokenCount = 0;
  clearInterval(state.timer);
  state.timer = setInterval(() => {
    $("elapsed").textContent = seconds((performance.now() - state.startedAt) / 1000);
  }, 80);
  $("tokenOutput").textContent = "";
  $("ttft").textContent = "-";
  $("tokenCount").textContent = "0";
  $("finishReason").textContent = "-";
  $("requestId").textContent = "-";
  $("queueSeconds").textContent = "-";
  $("batchId").textContent = "-";
  $("batchSize").textContent = "-";
  $("prefixRoute").textContent = "-";
  $("prefixMatched").textContent = "-";
  $("prefixRatio").textContent = "-";
  $("prefixSaved").textContent = "-";
  setStage("received");
}

function payload() {
  const messages = [];
  if ($("systemPrompt").value.trim()) {
    messages.push({role: "system", content: $("systemPrompt").value});
  }
  messages.push({role: "user", content: $("userPrompt").value});
  return {
    model: $("modelSelect").value,
    messages,
    max_tokens: Number($("maxTokens").value),
    temperature: Number($("temperature").value),
    stream: $("streamToggle").checked,
  };
}

function applyHeaders(response) {
  $("queueSeconds").textContent = seconds(response.headers.get("X-Orchard-Queue-Seconds"));
  $("batchId").textContent = response.headers.get("X-Orchard-Batch-ID") || "-";
  $("batchSize").textContent = response.headers.get("X-Orchard-Batch-Size") || "-";
  $("prefixRoute").textContent = response.headers.get("X-Orchard-Prefix-Route") || "-";
  $("prefixMatched").textContent = response.headers.get("X-Orchard-Prefix-Matched-Tokens") || "-";
  $("prefixSaved").textContent =
    response.headers.get("X-Orchard-Prefix-Estimated-Saved-Tokens") || "-";
}

function applyOrchard(meta) {
  if (!meta) return;
  $("queueSeconds").textContent = seconds(meta.queue_seconds);
  $("batchId").textContent = meta.batch_id || "-";
  $("batchSize").textContent = meta.batch_size ?? "-";
  $("prefixRoute").textContent = meta.prefix_route || "-";
  $("prefixMatched").textContent = meta.prefix_matched_tokens ?? "-";
  $("prefixRatio").textContent = meta.prefix_matched_ratio === undefined
    ? "-"
    : Number(meta.prefix_matched_ratio).toFixed(3);
  $("prefixSaved").textContent = meta.prefix_estimated_saved_tokens ?? "-";
}

function noteToken(text) {
  if (!state.firstTokenAt) {
    state.firstTokenAt = performance.now();
    $("ttft").textContent = seconds((state.firstTokenAt - state.startedAt) / 1000);
    setStage("decoding");
  }
  state.tokenCount += 1;
  $("tokenCount").textContent = String(state.tokenCount);
  $("tokenOutput").textContent += text;
}

async function sendRequest() {
  resetRun();
  $("sendBtn").disabled = true;
  try {
    setStage("validated");
    const body = payload();
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify(body),
    });
    applyHeaders(response);
    setStage("queued");
    if (!response.ok) {
      const err = await response.text();
      $("tokenOutput").textContent = err;
      setStage("completed", true);
      return;
    }
    $("requestId").textContent = response.headers.get("X-Request-ID") || "-";
    if (body.stream) {
      await consumeStream(response);
    } else {
      const data = await response.json();
      $("requestId").textContent = data.id;
      const content = data.choices?.[0]?.message?.content || "";
      noteToken(content);
      $("finishReason").textContent = data.choices?.[0]?.finish_reason || "-";
      applyOrchard(data.orchard);
    }
    setStage("completed");
  } catch (error) {
    $("tokenOutput").textContent = String(error);
    setStage("completed", true);
  } finally {
    clearInterval(state.timer);
    $("elapsed").textContent = seconds((performance.now() - state.startedAt) / 1000);
    $("sendBtn").disabled = false;
    await refreshMetrics();
  }
}

async function consumeStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    const chunks = buffer.split("\\n\\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const line = chunk.split("\\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      const raw = line.slice(6);
      if (raw === "[DONE]") continue;
      const event = JSON.parse(raw);
      if (event.id) $("requestId").textContent = event.id;
      if (event.error) {
        $("tokenOutput").textContent += `\\n[${event.error.type}] ${event.error.message}`;
        continue;
      }
      const delta = event.choices?.[0]?.delta || {};
      if (delta.content !== undefined) noteToken(delta.content);
      const finish = event.choices?.[0]?.finish_reason;
      if (finish) $("finishReason").textContent = finish;
      applyOrchard(event.orchard);
    }
  }
}

$("refreshBtn").addEventListener("click", refreshStatus);
$("sendBtn").addEventListener("click", sendRequest);
$("runChunkedBtn").addEventListener("click", runChunkedPrefill);
$("runKvBtn").addEventListener("click", runKvBlocks);
refreshStatus();
runChunkedPrefill();
runKvBlocks();
setInterval(refreshMetrics, 5000);
"""


@router.get("/ui", include_in_schema=False)
async def control_room() -> HTMLResponse:
    """Serve the local control-room UI."""

    return HTMLResponse(HTML)


@router.get("/ui/styles.css", include_in_schema=False)
async def styles() -> Response:
    """Serve UI CSS."""

    return Response(CSS, media_type="text/css")


@router.get("/ui/app.js", include_in_schema=False)
async def app_js() -> Response:
    """Serve UI JavaScript."""

    return Response(JS, media_type="application/javascript")


@router.post("/ui/simulate/chunked-prefill", include_in_schema=False)
async def simulate_chunked_prefill(body: ChunkedPrefillBody) -> dict[str, object]:
    """Run the chunked-prefill simulator for the UI."""

    specs = deterministic_workload(
        requests=body.requests,
        arrival_interval_steps=body.arrival_interval_steps,
        prompt_tokens=body.prompt_tokens,
        output_tokens=body.output_tokens,
        prefix_saved_tokens=body.prefix_saved_tokens,
    )
    runs = []
    for policy in ChunkedPrefillPolicy:
        result = ChunkedPrefillSimulator(
            chunk_size=body.chunk_size,
            policy=policy,
            decode_tpot_slo_steps=body.decode_tpot_slo_steps,
        ).run(specs)
        runs.append(
            {
                "policy": policy.value,
                "summary": summarize(result),
                "events": [
                    {
                        "step": event.step,
                        "operation": event.operation,
                        "request_id": event.request_id,
                        "tokens_processed": event.tokens_processed,
                        "waiting_prefill": event.waiting_prefill,
                        "waiting_decode": event.waiting_decode,
                    }
                    for event in result.events[:160]
                ],
            }
        )
    return {"runs": runs}


@router.post("/ui/simulate/kv-blocks", include_in_schema=False)
async def simulate_kv_blocks(body: KVBlocksBody) -> dict[str, object]:
    """Run the KV block manager simulator for the UI."""

    return deterministic_scenario(
        block_size_tokens=body.block_size_tokens,
        sequences=body.sequences,
        base_prompt_tokens=body.base_prompt_tokens,
        shared_prefix_tokens=body.shared_prefix_tokens,
        decode_tokens=body.decode_tokens,
    )
