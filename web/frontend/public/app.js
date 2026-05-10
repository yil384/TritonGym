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
  view: "flat",          // "flat" | "agent"
  sortKey: "pass",
  sortDir: "desc",
};

const AGENT_ORDER = ["AlphaEvolve", "Geak", "One-shot", "Leader"];

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
  const rows = data.rows.slice();

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty">No rows to show.</td></tr>`;
    return;
  }

  // Visual bar scaling uses the global max pass on the current split,
  // so bars stay comparable across the agent groups.
  const maxPass = rows.reduce((m, r) => Math.max(m, r.splits[state.split]?.pass || 0), 0) || 1;

  if (state.view === "agent") {
    // Group by agent, ordered by AGENT_ORDER. Within each group, sort by Pass desc.
    const groups = new Map();
    for (const r of rows) {
      const a = r.agent_display;
      if (!groups.has(a)) groups.set(a, []);
      groups.get(a).push(r);
    }
    const sortedAgents = [...groups.keys()].sort((a, b) => {
      const ia = AGENT_ORDER.indexOf(a), ib = AGENT_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });

    const html = sortedAgents.map(agent => {
      const grp = groups.get(agent).slice().sort(compareRows);
      const groupRows = grp.map((r, i) => renderRow(r, i + 1, maxPass, /*showAgent*/ false)).join("");
      return `<tr class="group-header"><td colspan="5">${escapeHtml(agent)} <span class="group-count">${grp.length} model${grp.length === 1 ? "" : "s"}</span></td></tr>${groupRows}`;
    }).join("");

    tbody.innerHTML = html;
  } else {
    // Flat: respect current sort.
    rows.sort(compareRows);
    tbody.innerHTML = rows.map((r, i) => renderRow(r, i + 1, maxPass, true)).join("");
  }
}

function renderRow(r, rank, maxPass, showAgent) {
  const s = r.splits[state.split] || {};
  const passWidth = ((s.pass || 0) / maxPass) * 100;
  return `<tr>
    <td class="col-rank">${rank}</td>
    <td class="col-model">${escapeHtml(r.model_display)}</td>
    <td class="col-agent">${showAgent ? escapeHtml(r.agent_display) : ""}</td>
    <td class="col-pass"><div class="bar-cell"><span>${fmtPct(s.pass)}</span><div class="bar"><div class="bar-fill" style="width:${passWidth}%"></div></div></div></td>
    <td class="col-perf">${fmtPerf(s.perf)}</td>
  </tr>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function bindControls() {
  // Split tabs (Full/Standard/OOD/DSL)
  $$("#split-tabs .tab").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#split-tabs .tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.split = btn.dataset.split;
      render();
    });
  });

  // View tabs (Flat / By Agent)
  $$("#view-tabs .tab").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#view-tabs .tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.view = btn.dataset.view;
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
    $("#lb-body").innerHTML = `<tr><td colspan="5" class="empty">Could not load leaderboard data: ${escapeHtml(e.message)}</td></tr>`;
  }
})();
