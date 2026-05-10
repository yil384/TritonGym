// TritonGym leaderboard frontend.
// Reads /api/leaderboard (or /data/leaderboard.json as fallback) and renders
// a sortable, splitable, filterable table.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const SPLIT_LABELS = {
  full:     "Full Benchmark",
  standard: "Standard",
  ood:      "OOD",
  dsl:      "DSL",
};

const state = {
  data: null,
  split: "full",
  sortKey: "pass",
  sortDir: "desc",
  showPaper: true,
  showLogs: true,
};

async function loadData() {
  // Prefer the API (when served behind FastAPI) but fall back to the static
  // JSON when the user opens index.html directly.
  for (const url of ["/api/leaderboard", "data/leaderboard.json"]) {
    try {
      const res = await fetch(url);
      if (res.ok) return await res.json();
    } catch (_) { /* try next */ }
  }
  throw new Error("Could not load leaderboard data");
}

function fmtPct(p) {
  if (p == null || isNaN(p)) return "—";
  return (p * 100).toFixed(1) + "%";
}
function fmtPerf(p) {
  if (p == null || isNaN(p)) return "—";
  return p.toFixed(3);
}

function rowSortValue(row, key) {
  if (key === "model") return row.model_display;
  if (key === "agent") return row.agent_display;
  const split = row.splits[state.split] || {};
  if (key === "pass") return split.pass ?? -1;
  if (key === "perf") return split.perf ?? -1;
  return 0;
}

function compareRows(a, b) {
  const ka = rowSortValue(a, state.sortKey);
  const kb = rowSortValue(b, state.sortKey);
  let cmp;
  if (typeof ka === "number" && typeof kb === "number") cmp = ka - kb;
  else cmp = String(ka).localeCompare(String(kb));
  return state.sortDir === "asc" ? cmp : -cmp;
}

function render() {
  const data = state.data;
  if (!data) return;

  const tbody = $("#lb-body");
  const filtered = data.rows.filter(r => {
    const src = r.source || (r.tokens && (r.tokens.input || r.tokens.output) ? "logs" : "logs");
    if (src === "paper" && !state.showPaper) return false;
    if (src === "logs"  && !state.showLogs)  return false;
    return true;
  });

  filtered.sort(compareRows);

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty">No rows match your filters.</td></tr>`;
    return;
  }

  // For visual scaling of the bar cell, use the max pass on the current split.
  const maxPass = filtered.reduce((m, r) => Math.max(m, r.splits[state.split]?.pass || 0), 0) || 1;

  tbody.innerHTML = filtered.map((r, i) => {
    const s = r.splits[state.split] || {};
    const cov = s.attempted != null && s.total
      ? `${s.attempted}/${s.total}`
      : (s.total ? `${s.total}/${s.total}` : "—");
    const passWidth = ((s.pass || 0) / maxPass) * 100;
    const src = r.source || "logs";
    return `<tr>
      <td class="col-rank">${i + 1}</td>
      <td class="col-model">${escapeHtml(r.model_display)}</td>
      <td class="col-agent">${escapeHtml(r.agent_display)}</td>
      <td class="col-pass"><div class="bar-cell"><span>${fmtPct(s.pass)}</span><div class="bar"><div class="bar-fill" style="width:${passWidth}%"></div></div></div></td>
      <td class="col-perf">${fmtPerf(s.perf)}</td>
      <td class="col-cov">${cov}</td>
      <td class="col-src"><span class="tag tag-${src}">${src}</span></td>
    </tr>`;
  }).join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function bindControls() {
  $$(".tabs .tab").forEach(btn => {
    btn.addEventListener("click", () => {
      $$(".tabs .tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.split = btn.dataset.split;
      render();
    });
  });

  $$(".sortable").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (state.sortKey === key) {
        state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
      } else {
        state.sortKey = key;
        state.sortDir = key === "pass" || key === "perf" ? "desc" : "asc";
      }
      $$(".sortable").forEach(t => {
        t.classList.remove("sort-active", "sort-desc", "sort-asc");
        const baseLabel = t.textContent.replace(/[▲▼]\s*$/, "").trim();
        t.textContent = baseLabel;
        if (t.dataset.sort === state.sortKey) {
          t.classList.add("sort-active", "sort-" + state.sortDir);
          t.textContent = baseLabel + " " + (state.sortDir === "desc" ? "▼" : "▲");
        }
      });
      render();
    });
  });

  $("#filter-paper").addEventListener("change", e => { state.showPaper = e.target.checked; render(); });
  $("#filter-logs").addEventListener("change",  e => { state.showLogs  = e.target.checked; render(); });
}

function setHeroStats(data) {
  const totalOps = (data.splits.standard?.size || 0) + (data.splits.ood?.size || 0) + (data.splits.dsl?.size || 0);
  $("#stat-ops").textContent = totalOps;
  const models = new Set(data.rows.map(r => r.model_display));
  $("#stat-models").textContent = models.size;
}

(async function init() {
  bindControls();
  try {
    state.data = await loadData();
    setHeroStats(state.data);
    render();
  } catch (e) {
    $("#lb-body").innerHTML = `<tr><td colspan="7" class="empty">Could not load leaderboard data: ${escapeHtml(e.message)}</td></tr>`;
  }
})();
