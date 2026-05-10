// Submission form — POST /api/submissions, plus poll the queue status.

const form = document.getElementById("submit-form");
const resultBox = document.getElementById("submit-result");
const submitBtn = document.getElementById("submit-btn");

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  resultBox.className = "submit-result";
  resultBox.textContent = "";
  submitBtn.disabled = true;

  const data = new FormData(form);
  const agents = data.getAll("agents");
  const payload = {
    hf_model: (data.get("hf_model") || "").trim(),
    email:    (data.get("email")    || "").trim(),
    notes:    (data.get("notes")    || "").trim(),
    agents,
  };

  try {
    const res = await fetch("/api/submissions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = body.detail || `HTTP ${res.status}`;
      resultBox.classList.add("submit-result-error");
      resultBox.textContent = "Error: " + detail;
    } else {
      resultBox.classList.add("submit-result-ok");
      resultBox.textContent = `Submission #${body.id} queued. ${body.message || ""}`;
      form.reset();
      refreshQueue();
    }
  } catch (e) {
    resultBox.classList.add("submit-result-error");
    resultBox.textContent = "Network error: " + e.message;
  } finally {
    submitBtn.disabled = false;
  }
});

async function refreshQueue() {
  try {
    const res = await fetch("/api/submissions/queue");
    if (!res.ok) return;
    const { pending, completed } = await res.json();
    document.getElementById("qs-pending").textContent   = pending;
    document.getElementById("qs-completed").textContent = completed;
  } catch (_) { /* ignore */ }
}
refreshQueue();
