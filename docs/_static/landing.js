const root = document.querySelector(".bd-landing");
if (root) {
  const copyTimers = new WeakMap();
  let lastMsg = "";
  root.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    const status = btn.parentElement?.querySelector(".bd-copy-status");
    clearTimeout(copyTimers.get(btn));
    let ok = false;
    try { await navigator.clipboard.writeText(btn.dataset.copy); ok = true; } catch {}
    btn.toggleAttribute("data-copied", ok);
    if (status) {
      let msg = ok ? "Copied to clipboard" : "Copy failed";
      if (msg === lastMsg) msg += "\u00A0";
      lastMsg = msg;
      status.textContent = msg;
    }
    copyTimers.set(btn, setTimeout(() => { btn.removeAttribute("data-copied"); if (status) status.textContent = ""; }, 1800));
  });

  const zoo = root.querySelector(".bd-zoo");
  if (zoo) {
    const cards = [...zoo.querySelectorAll(".zoo-card")];
    const searchInput = zoo.querySelector("#bd-zoo-q");
    const moreBox = zoo.querySelector("#bd-zoo-more");
    const countOut = zoo.querySelector(".zoo-foot-count");
    const syncZoo = () => {
      const q = (searchInput?.value || "").trim().toLowerCase();
      const mod = zoo.querySelector('input[name="bd-zoo-mod"]:checked')?.value || "all";
      const cat = zoo.querySelector('input[name="bd-zoo-cat"]:checked')?.value || "all";
      const expanded = Boolean(q) || mod !== "all" || cat !== "all" || Boolean(moreBox?.checked);
      zoo.toggleAttribute("data-searching", Boolean(q));
      let matched = 0;
      for (const card of cards) {
        const hitQ = !q || card.dataset.search.includes(q);
        card.hidden = !hitQ;
        if (hitQ && (mod === "all" || card.dataset.mod.split(" ").includes(mod)) && (cat === "all" || card.dataset.cats.split(" ").includes(cat))) matched++;
      }
      zoo.toggleAttribute("data-empty", matched === 0);
      if (countOut) countOut.textContent = expanded ? `Showing ${matched} of ${cards.length} architectures` : `Showing 12 featured of ${cards.length} architectures`;
    };
    searchInput?.addEventListener("input", syncZoo);
    zoo.addEventListener("change", syncZoo);
    document.addEventListener("keydown", (e) => {
      if (e.key === "/" && !e.metaKey && !e.ctrlKey && !e.altKey && !e.target?.matches?.("input, textarea, select, [contenteditable]")) {
        e.preventDefault();
        searchInput?.focus();
      }
    });
  }

  const dlEl = root.querySelector("#bd-downloads-num");
  if (dlEl) {
    const io = new IntersectionObserver((entries, obs) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      obs.disconnect();
      fetch("https://img.shields.io/pepy/dt/braindecode.json", { priority: "low" })
        .then((r) => (r.ok ? r.json() : null))
        .then((j) => { if (j?.value) dlEl.textContent = String(j.value).replace(/k$/, "K").replace(/m$/, "M"); })
        .catch(() => {});
    }, { rootMargin: "200px" });
    io.observe(dlEl);
  }
}
