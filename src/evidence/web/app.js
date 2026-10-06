let page = "search",
  docs = [];
const tabs = [
  ["search", "Search evidence"],
  ["documents", "Document library"],
  ["graph", "Knowledge paths"],
  ["benchmark", "Retrieval benchmark"],
  ["activity", "Query activity"],
];
function heading(t, d, a = "") {
  return `<div class="page-title"><div><h1>${t}</h1><p class="muted">${d}</p></div>${a}</div>`;
}
async function render(next = page) {
  page = next;
  setNav(tabs, page, guarded(render));
  $("#main").innerHTML = '<div class="loading">Loading evidence…</div>';
  docs = (await api("/documents")).items;
  if (page === "search") search();
  if (page === "documents") library();
  if (page === "graph") graph();
  if (page === "benchmark") await benchmark();
  if (page === "activity") await activity();
}
function search() {
  $("#main").innerHTML =
    heading(
      "Every answer begins with a source.",
      "Search the current documents your identity can access, then inspect the exact evidence.",
    ) +
    `<div class="stats">${stat("Accessible documents", docs.length, "Current tenant and role")}${stat("Retrieval methods", config.neural ? "4" : "1", config.neural ? "Lexical · dense · fusion · rerank" : "BM25 lexical retrieval")}${stat("Citation format", "Versioned", "Document / revision / chunk")}${stat("Default response", "Excerpts", "Source relevance is not entailment")}</div><div class="card"><div class="toolbar"><input id="question" aria-label="Evidence question" placeholder="How should we handle a currency mismatch?" value="How should we handle a currency mismatch?"><select id="method" aria-label="Retrieval method"><option value="bm25">BM25 · lexical</option>${config.neural ? '<option value="dense">MiniLM · dense</option><option value="fusion" selected>Hybrid · rank fusion</option><option value="rerank">Hybrid + reranker</option>' : ""}</select><button class="primary" id="search">Find evidence</button></div><div class="row small muted"><span>Try:</span>${["What happens when a worker crashes?", "Who can approve a case?", "How is DV01 calculated?"].map((q) => `<button class="small" data-question="${esc(q)}">${esc(q)}</button>`).join("")}</div></div><div id="results" style="margin-top:22px"></div>`;
  $("#question").onkeydown = (e) => {
    if (e.key === "Enter") $("#search").click();
  };
  document.querySelectorAll("[data-question]").forEach(
    (b) =>
      (b.onclick = () => {
        $("#question").value = b.dataset.question;
        $("#search").click();
      }),
  );
  $("#search").onclick = guarded(async () => {
    let button = $("#search");
    button.disabled = true;
    $("#results").innerHTML =
      '<div class="loading">Retrieving authorized evidence…</div>';
    try {
      let r = await post("/search", {
        question: $("#question").value,
        method: $("#method").value,
        limit: 5,
      });
      $("#results").innerHTML =
        `<div class="row spread" style="margin-bottom:15px"><h2>${r.hits.length} evidence passages</h2><span class="small muted">${fmt(r.latency_ms, 1)} ms · ${esc(r.method)}</span></div>${r.hits.map((h, i) => `<article class="card" style="margin-bottom:15px"><div class="row spread"><h2>${i + 1}. ${esc(h.title)}</h2><span class="badge">revision ${h.revision}</span></div><div class="quote">${esc(h.text)}</div><div class="row spread"><span class="mono small muted">${esc(h.id)}</span><span class="small muted">Score ${fmt(h.score, 4)}${h.dense_score != null ? " · cosine " + fmt(h.dense_score, 3) : ""}</span></div><details><summary>Source integrity and scoring</summary><p class="small">SHA-256: <span class="mono">${esc(h.sha256)}</span></p><p class="small">Lexical score: ${fmt(h.lexical_score, 3)}. Scores rank passages; they are not probabilities that an answer is correct.</p><button data-open="${esc(h.document_id)}">Open source document</button></details></article>`).join("")}${!r.hits.length ? '<div class="card">' + empty("No accessible evidence matched. Try another query or add a document.") + "</div>" : ""}<div class="card"><div class="row"><strong class="small">Were these passages useful?</strong><button id="helpful">Useful</button><button id="not-helpful">Needs work</button><input id="feedback-note" placeholder="Optional feedback" aria-label="Retrieval feedback"></div></div>`;
      for (let [id, useful] of [
        ["helpful", true],
        ["not-helpful", false],
      ])
        $("#" + id).onclick = guarded(async () => {
          await post("/feedback/" + r.query_id, {
            useful,
            note: $("#feedback-note").value,
          });
          toast("Feedback saved");
        });
      document.querySelectorAll("[data-open]").forEach(
        (b) =>
          (b.onclick = guarded(async () => {
            await render("documents");
            await editDocument(b.dataset.open);
          })),
      );
    } finally {
      button.disabled = false;
    }
  });
}
function library() {
  $("#main").innerHTML =
    heading(
      "An evolving, governed library.",
      "Revisions are immutable. Current permissions are checked before retrieval.",
      '<button class="primary" id="new">Add document</button>',
    ) +
    `<div class="split"><section class="card flush"><div class="table-wrap"><table><thead><tr><th>Document</th><th>Access</th><th>Revision</th></tr></thead><tbody>${docs.map((d) => `<tr class="clickable" data-doc="${esc(d.id)}"><td><strong>${esc(d.title)}</strong><div class="mono small muted">${esc(d.id)}</div></td><td>${d.roles.map((r) => '<span class="pill">' + esc(r) + "</span>").join("")}</td><td>${d.revision}</td></tr>`).join("")}</tbody></table></div>${!docs.length ? empty("No documents are visible to this identity.") : ""}</section><section class="card" id="editor">${empty("Select a document to inspect, revise, or revoke access.")}</section></div>`;
  document
    .querySelectorAll("[data-doc]")
    .forEach(
      (el) => (el.onclick = guarded(() => editDocument(el.dataset.doc))),
    );
  $("#new").onclick = guarded(() => editDocument(null));
}
async function editDocument(id) {
  let d = docs.find((x) => x.id === id) || {
    id: "",
    title: "",
    text: "",
    roles: ["analyst", "reviewer"],
    edges: [],
  };
  let history = id
    ? (await api("/documents/" + encodeURIComponent(id) + "/history")).items
    : [];
  $("#editor").innerHTML =
    `<h2>${id ? "Document details" : "New document"}</h2><div class="form-grid"><div><label for="doc-id">Stable identifier</label><input id="doc-id" value="${esc(d.id)}" ${id ? "readonly" : ""} placeholder="policy-name"></div><div><label for="doc-access">Allowed roles</label><select id="doc-access"><option value="both">Analyst and reviewer</option><option value="reviewer" ${d.roles.length === 1 && d.roles[0] === "reviewer" ? "selected" : ""}>Reviewer only</option></select></div></div><label for="doc-title" style="margin-top:15px">Title</label><input id="doc-title" style="width:100%" value="${esc(d.title)}"><label for="doc-text" style="margin-top:15px">Document text</label><textarea id="doc-text" style="min-height:240px">${esc(d.text)}</textarea><div class="row" style="margin-top:15px"><button class="primary" id="save">Save revision</button>${id ? '<button class="danger" id="delete">Delete document</button>' : ""}</div><p class="small muted">Ingestion and deletion require the reviewer role. Removing analyst access takes effect on the next query, including cached retrieval snapshots.</p>${history.length ? `<h3>Revision history</h3>${history.map((h) => `<details><summary>Revision ${h.revision} · ${esc(h.roles.join(", "))}</summary><p class="small">${esc(h.text)}</p><p class="small mono" style="overflow-wrap:anywhere">${esc(h.fingerprint)}</p></details>`).join("")}` : ""}`;
  $("#save").onclick = guarded(async () => {
    let result = await post("/documents", {
      id: $("#doc-id").value,
      title: $("#doc-title").value,
      text: $("#doc-text").value,
      roles:
        $("#doc-access").value === "both"
          ? ["analyst", "reviewer"]
          : ["reviewer"],
      edges: d.edges,
    });
    toast("Saved revision " + result.revision);
    let current = $("#doc-id").value;
    await render("documents");
    if (docs.some((x) => x.id === current)) await editDocument(current);
  });
  if ($("#delete"))
    $("#delete").onclick = guarded(async () => {
      await api("/documents/" + encodeURIComponent(id), { method: "DELETE" });
      toast("Document deleted; retrieval access revoked");
      await render("documents");
    });
}
function graph() {
  let nodes = [
    ...new Set(
      docs.flatMap((d) => d.edges.flatMap((e) => [e.source, e.target])),
    ),
  ].sort();
  $("#main").innerHTML =
    heading(
      "Follow a claim to its owner.",
      "Traverse curated relationships with document and revision provenance on every edge.",
    ) +
    `<div class="card"><div class="toolbar"><select id="from" aria-label="Graph source">${nodes.map((n) => `<option ${n === "quantity-break" ? "selected" : ""}>${esc(n)}</option>`).join("")}</select><span>→</span><select id="to" aria-label="Graph target">${nodes.map((n) => `<option ${n === "desk-supervisor" ? "selected" : ""}>${esc(n)}</option>`).join("")}</select><button class="primary" id="trace">Trace relationship</button></div><div id="path"></div></div><div class="notice">This graph contains explicitly curated relationships. The service does not infer missing edges or treat a connected path as proof of a financial claim.</div><div class="card"><h2>Visible relationships</h2><table><thead><tr><th>From</th><th>Relationship</th><th>To</th><th>Source</th></tr></thead><tbody>${docs.flatMap((d) => d.edges.map((e) => `<tr><td class="mono">${esc(e.source)}</td><td>${esc(e.relation)}</td><td class="mono">${esc(e.target)}</td><td>${esc(d.title)} · v${d.revision}</td></tr>`)).join("")}</tbody></table></div>`;
  $("#trace").onclick = guarded(async () => {
    let r = await post("/graph/path", {
      source: $("#from").value,
      target: $("#to").value,
      max_hops: 5,
    });
    $("#path").innerHTML = r.path
      ? '<div class="timeline">' +
        r.path
          .map(
            (e) =>
              `<div class="event"><strong>${esc(e.source)}</strong><p>${esc(e.relation)} → <strong>${esc(e.target)}</strong></p><span class="badge">${esc(e.document_id)} · revision ${e.revision}</span></div>`,
          )
          .join("") +
        "</div>"
      : empty("No authorized path connects these nodes within five hops.");
  });
  $("#trace").click();
}
async function benchmark() {
  let d = await api("/benchmarks/scifact"),
    s = d.summary;
  $("#main").innerHTML =
    heading(
      "Measure the retrieval tradeoff.",
      "Pinned models, the complete corpus, and every official test query. No SciFact label fitting.",
    ) +
    `<div class="stats">${stat("Corpus", fmt(d.config.corpus_documents), "BEIR SciFact documents")}${stat("Test queries", d.config.test_queries, "Full official test split")}${stat("Best nDCG@10", fmt(Math.max(...Object.values(s).map((v) => v.ndcg_at_10)), 3), "Higher is better")}${stat("Candidate pool", "50", "Before cross-encoder reranking")}</div><div class="card flush"><table><thead><tr><th>Method</th><th>nDCG@10</th><th>Recall@10</th><th>MRR@10</th><th>p50 latency</th><th>p95 latency</th></tr></thead><tbody>${Object.entries(
      s,
    )
      .map(
        ([n, v]) =>
          `<tr><td><strong>${esc(n)}</strong></td><td>${fmt(v.ndcg_at_10, 4)}<div class="bar"><span style="width:${v.ndcg_at_10 * 100}%"></span></div><span class="small muted">95% CI ${v.ndcg_ci95.map((x) => fmt(x, 3)).join("–")}</span></td><td>${pct(v.recall_at_10)}</td><td>${fmt(v.mrr_at_10, 4)}</td><td>${fmt(v.p50_ms, 1)} ms</td><td>${fmt(v.p95_ms, 1)} ms</td></tr>`,
      )
      .join(
        "",
      )}</tbody></table></div><div class="grid two" style="margin-top:20px"><section class="card"><h2>What the experiment says</h2><p>Rank fusion improves retrieval on this corpus. The small cross-encoder adds latency and does not outperform fusion, so it is an optional method.</p><p class="muted small">This measures document relevance on scientific abstracts. It does not establish financial-domain accuracy or answer faithfulness. Timings exclude initial model loading and index construction.</p></section><section class="card"><h2>Reproduce the run</h2><pre class="code">pip install -e '.[research]'
python -m evidence.benchmark \\
  --output runtime/scifact-study</pre><details><summary>Protocol and pinned model revisions</summary>${jsonView(d.config)}</details></section></div>`;
}
async function activity() {
  let d = await api("/activity");
  $("#main").innerHTML =
    heading(
      "Inspect the retrieval trail.",
      "Your queries are logged as hashes with citation IDs and timing. Raw query text is not retained.",
    ) +
    `<div class="card flush"><table><thead><tr><th>Time</th><th>Method</th><th>Passages</th><th>Latency</th><th>Query fingerprint</th></tr></thead><tbody>${d.items.map((r) => `<tr><td>${when(r.created_at)}</td><td>${esc(r.method)}</td><td>${r.citations.length}</td><td>${fmt(r.latency_ms, 1)} ms</td><td class="mono small">${esc(r.query_sha256.slice(0, 20))}…</td></tr>`).join("")}</tbody></table>${!d.items.length ? empty("Run a search to create the first retrieval event.") : ""}</div>`;
}
window.addEventListener(
  "identity-changed",
  guarded(() => render()),
);
guarded(async () => {
  await initAuth();
  await render();
})();
