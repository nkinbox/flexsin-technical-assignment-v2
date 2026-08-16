/* Document-Aware AI Chatbot -- frontend.
   No build step and no framework: this is a Python project, and adding a Node
   toolchain would buy nothing for a single page. */

const $ = (id) => document.getElementById(id);

const state = {
  sessionId: null,
  documents: [],
  selected: new Set(),   // empty = search everything (multi-document default)
  citationsByMessage: new Map(),
  busy: false,
};

/* ------------------------------------------------------------ helpers */

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const formatBytes = (n) => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ------------------------------------------------------------- health */

async function refreshHealth() {
  try {
    const health = await api("/health");
    const ready = health.weaviate_ready;
    $("statusDot").className = `dot ${ready ? "ok" : "bad"}`;
    $("statusText").textContent = ready
      ? `${health.provider} · ready`
      : `${health.provider} · vector store offline`;
    if (health.provider === "dev") {
      $("status").title =
        "PROVIDER=dev: the pipeline is real, but embeddings are stubs and " +
        "answers are templates. Set PROVIDER=vertex for real answers.";
    }
  } catch {
    $("statusDot").className = "dot bad";
    $("statusText").textContent = "backend unreachable";
  }
}

/* ---------------------------------------------------------- documents */

async function loadDocuments() {
  try {
    const data = await api("/documents");
    state.documents = data.documents || [];
  } catch {
    state.documents = [];
  }
  renderDocuments();
}

function renderDocuments() {
  const list = $("docList");
  list.innerHTML = "";

  $("docCount").textContent =
    `${state.documents.length} document${state.documents.length === 1 ? "" : "s"}`;

  if (!state.documents.length) {
    list.innerHTML =
      `<li class="doc-sub" style="padding:8px 4px">Nothing uploaded yet.</li>`;
    updateScopeNote();
    return;
  }

  for (const doc of state.documents) {
    const li = document.createElement("li");
    li.className = "doc-item";

    const visionBadge = doc.vision_pages?.length
      ? `<span class="badge-vision" title="Extracted with Gemini vision">vision ×${doc.vision_pages.length}</span>`
      : "";

    li.innerHTML = `
      <input type="checkbox" data-id="${doc.doc_id}"
             ${state.selected.has(doc.doc_id) ? "checked" : ""} />
      <div class="doc-info">
        <div class="doc-name">${escapeHtml(doc.doc_name)}${visionBadge}</div>
        <div class="doc-sub">
          ${doc.chunk_count} chunks · ${doc.page_count} page${doc.page_count === 1 ? "" : "s"}
          · ${formatBytes(doc.size_bytes)}
        </div>
      </div>
      <button class="icon-btn" data-delete="${doc.doc_id}" title="Delete">×</button>`;

    li.querySelector("input").addEventListener("change", (event) => {
      if (event.target.checked) state.selected.add(doc.doc_id);
      else state.selected.delete(doc.doc_id);
      updateScopeNote();
    });

    li.querySelector("[data-delete]").addEventListener("click", async () => {
      if (!confirm(`Delete "${doc.doc_name}" and all of its chunks?`)) return;
      try {
        await api(`/documents/${doc.doc_id}`, { method: "DELETE" });
        state.selected.delete(doc.doc_id);
        await loadDocuments();
      } catch (error) {
        alert(`Delete failed: ${error.message}`);
      }
    });

    list.appendChild(li);
  }
  updateScopeNote();
}

function updateScopeNote() {
  const n = state.selected.size;
  $("scopeNote").textContent = n === 0
    ? "Searching all documents"
    : `Searching ${n} selected document${n === 1 ? "" : "s"}`;
}

/* ------------------------------------------------------------ uploads */

async function uploadFiles(files) {
  const queue = $("uploadQueue");

  for (const file of files) {
    const row = document.createElement("div");
    row.className = "upload-item";
    row.innerHTML =
      `<span>${escapeHtml(file.name)}</span><span class="spinner"></span>`;
    queue.appendChild(row);

    const form = new FormData();
    form.append("file", file);

    try {
      const meta = await api("/documents", { method: "POST", body: form });
      row.innerHTML =
        `<span>${escapeHtml(file.name)}</span><span>${meta.chunk_count} chunks</span>`;
      setTimeout(() => row.remove(), 2500);
      await loadDocuments();
    } catch (error) {
      row.className = "upload-item error";
      row.innerHTML =
        `<span>${escapeHtml(file.name)}</span><span>${escapeHtml(error.message)}</span>`;
      setTimeout(() => row.remove(), 9000);
    }
  }
}

/* --------------------------------------------------------------- chat */

function addMessage(role, html) {
  const empty = $("messages").querySelector(".empty-state");
  if (empty) empty.remove();

  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  wrapper.innerHTML = `
    <div class="avatar">${role === "user" ? "You" : "AI"}</div>
    <div class="body">${html}</div>`;
  $("messages").appendChild(wrapper);
  $("messages").scrollTop = $("messages").scrollHeight;
  return wrapper;
}

/** Turn inline [n] markers into clickable chips. */
function renderAnswer(answer, messageId) {
  return escapeHtml(answer).replace(
    /\[(\d+)\]/g,
    (_, n) => `<span class="cite" data-msg="${messageId}" data-n="${n}">${n}</span>`
  );
}

function renderMeta(payload) {
  const bits = [];
  bits.push(payload.grounded
    ? `<span class="pill grounded">grounded</span>`
    : `<span class="pill ungrounded">not grounded</span>`);

  if (payload.grounded) {
    bits.push(`<span class="pill neutral">confidence ${payload.confidence}</span>`);
  }
  bits.push(`<span class="pill neutral">${payload.retrieved_count} passage${payload.retrieved_count === 1 ? "" : "s"}</span>`);

  if (payload.search_query && payload.search_query !== payload.question) {
    bits.push(`<span class="pill neutral" title="Your follow-up was rewritten into a standalone query before searching">rewritten: ${escapeHtml(payload.search_query)}</span>`);
  }
  if (payload.stripped_citations?.length) {
    bits.push(`<span class="pill ungrounded" title="The model cited passages that were not supplied; the markers were removed">${payload.stripped_citations.length} invented citation(s) stripped</span>`);
  }
  return `<div class="answer-meta">${bits.join("")}</div>`;
}

function renderSources(citations, messageId) {
  if (!citations.length) return "";
  const rows = citations.map((c) => `
    <div class="source-row" data-msg="${messageId}" data-n="${c.n}">
      <span class="n">[${c.n}]</span>
      <span>${escapeHtml(c.doc_name)} · ${escapeHtml(pageLabel(c))}</span>
      <span class="path">${escapeHtml(c.heading_path || "")}</span>
    </div>`).join("");
  return `<div class="sources">${rows}</div>`;
}

const pageLabel = (c) =>
  c.page_start === c.page_end ? `p.${c.page_start}` : `pp.${c.page_start}-${c.page_end}`;

async function ask(question) {
  if (state.busy) return;
  state.busy = true;
  $("sendBtn").disabled = true;

  addMessage("user", `<div class="text">${escapeHtml(question)}</div>`);
  const pending = addMessage(
    "bot", `<div class="typing"><span></span><span></span><span></span></div>`);

  try {
    const payload = await api("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        session_id: state.sessionId,
        doc_ids: state.selected.size ? [...state.selected] : null,
      }),
    });

    state.sessionId = payload.session_id;
    $("sessionLabel").textContent = `session ${payload.session_id.slice(0, 8)}`;

    const messageId = `m${state.citationsByMessage.size}`;
    state.citationsByMessage.set(messageId, payload.citations);

    pending.querySelector(".body").innerHTML =
      `<div class="text">${renderAnswer(payload.answer, messageId)}</div>`
      + renderMeta(payload)
      + renderSources(payload.citations, messageId);
  } catch (error) {
    pending.querySelector(".body").innerHTML =
      `<div class="text" style="color:var(--bad)">Request failed: ${escapeHtml(error.message)}</div>`;
  } finally {
    state.busy = false;
    $("sendBtn").disabled = false;
    $("messages").scrollTop = $("messages").scrollHeight;
  }
}

/* ------------------------------------------------------------- drawer */

function openDrawer(messageId, n) {
  const citations = state.citationsByMessage.get(messageId) || [];
  const citation = citations.find((c) => String(c.n) === String(n));
  if (!citation) return;

  $("drawerTitle").textContent = citation.doc_name;
  $("drawerMeta").textContent =
    [pageLabel(citation), citation.heading_path].filter(Boolean).join(" · ");

  // Highlight the span that actually matched, inside the wider parent section.
  const text = citation.chunk_text || "";
  const start = Math.max(0, Math.min(citation.highlight_start ?? 0, text.length));
  const end = Math.max(start, Math.min(citation.highlight_end ?? 0, text.length));

  $("drawerBody").innerHTML = end > start
    ? escapeHtml(text.slice(0, start))
      + `<mark>${escapeHtml(text.slice(start, end))}</mark>`
      + escapeHtml(text.slice(end))
    : escapeHtml(text);

  $("drawer").hidden = false;
  $("drawerBackdrop").hidden = false;
}

function closeDrawer() {
  $("drawer").hidden = true;
  $("drawerBackdrop").hidden = true;
}

/* -------------------------------------------------------------- wiring */

$("dropzone").addEventListener("click", () => $("fileInput").click());
$("dropzone").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("fileInput").click(); }
});
$("fileInput").addEventListener("change", (e) => {
  uploadFiles([...e.target.files]);
  e.target.value = "";
});

["dragenter", "dragover"].forEach((type) =>
  $("dropzone").addEventListener(type, (e) => {
    e.preventDefault();
    $("dropzone").classList.add("dragging");
  }));
["dragleave", "drop"].forEach((type) =>
  $("dropzone").addEventListener(type, (e) => {
    e.preventDefault();
    $("dropzone").classList.remove("dragging");
  }));
$("dropzone").addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));

$("selectAll").addEventListener("click", () => {
  if (state.selected.size === state.documents.length) state.selected.clear();
  else state.documents.forEach((d) => state.selected.add(d.doc_id));
  renderDocuments();
});

$("chatForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const value = $("question").value.trim();
  if (!value) return;
  $("question").value = "";
  $("question").style.height = "auto";
  ask(value);
});

$("question").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = `${Math.min(e.target.scrollHeight, 150)}px`;
});
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("chatForm").requestSubmit();
  }
});

$("newChat").addEventListener("click", () => {
  state.sessionId = null;
  state.citationsByMessage.clear();
  $("sessionLabel").textContent = "";
  $("messages").innerHTML = `
    <div class="empty-state">
      <h3>Ask a question about your documents</h3>
      <p>Answers come only from what you upload. If the documents don't
         contain the answer, the assistant says so instead of guessing.</p>
    </div>`;
});

// Citation chips and source rows are added dynamically, so delegate.
document.addEventListener("click", (e) => {
  const target = e.target.closest("[data-n][data-msg]");
  if (target) openDrawer(target.dataset.msg, target.dataset.n);
});
$("drawerClose").addEventListener("click", closeDrawer);
$("drawerBackdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

refreshHealth();
loadDocuments();
