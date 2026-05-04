(function () {
  "use strict";

  // Model zoo data: prefer the build-time auto-generated list (sourced from
  // braindecode/models/summary.csv via docs/sphinxext/zoo_data_gen.py).
  // Fall back to the inline curated list so this file still works when
  // served standalone. Each accessor lives in its own function to keep
  // the IIFE body's cyclomatic complexity below Codacy's 4 limit (the
  // `&&`/`||` short-circuits would otherwise inflate it).
  function injectedModels() {
    if (typeof window === "undefined") { return null; }
    return window.BD_MODELS || null;
  }
  function injectedCategories() {
    if (typeof window === "undefined") { return null; }
    return window.BD_CATEGORIES || null;
  }

  var BD_MODELS = injectedModels() || [
    { name: "EEGNetv4",          cat: "convolution",   year: 2018, params: "3.7 K",  paper: "Lawhern et al., 2018",          desc: "Compact depthwise + separable conv classifier" },
    { name: "ShallowFBCSPNet",   cat: "convolution",   year: 2017, params: "47 K",   paper: "Schirrmeister et al., 2017",    desc: "Filter-bank-CSP-style shallow ConvNet" },
    { name: "Deep4Net",          cat: "convolution",   year: 2017, params: "284 K",  paper: "Schirrmeister et al., 2017",    desc: "Four-block deep ConvNet for raw EEG" },
    { name: "EEGNeX",            cat: "convolution",   year: 2024, params: "60 K",   paper: "Chen et al., 2024",             desc: "Modernised EEGNet with dilations" },
    { name: "EEGInceptionERP",   cat: "convolution",   year: 2020, params: "14 K",   paper: "Santamaría-Vázquez et al.",     desc: "Inception-style ERP decoder" },
    { name: "EEGITNet",          cat: "convolution",   year: 2022, params: "9 K",    paper: "Salami et al., 2022",           desc: "Inception + temporal conv hybrid" },
    { name: "TIDNet",            cat: "convolution",   year: 2020, params: "212 K",  paper: "Kostas et al., 2020",           desc: "Temporal-inception decoder" },
    { name: "USleep",            cat: "convolution",   year: 2021, params: "3.1 M",  paper: "Perslev et al., 2021",          desc: "U-Net for sleep staging" },
    { name: "SleepStagerChambon",cat: "convolution",   year: 2018, params: "20 K",   paper: "Chambon et al., 2018",          desc: "Multi-channel sleep stager" },
    { name: "EEGConformer",      cat: "attention",     year: 2022, params: "871 K",  paper: "Song et al., 2022",             desc: "Conv front + transformer encoder" },
    { name: "ATCNet",            cat: "attention",     year: 2023, params: "114 K",  paper: "Altaheri et al., 2023",         desc: "Attention-temporal convolution net" },
    { name: "AttnBaseNet",       cat: "attention",     year: 2023, params: "98 K",   paper: "Wimpff et al., 2023",           desc: "Attention baseline for MI decoding" },
    { name: "CTNet",             cat: "attention",     year: 2024, params: "640 K",  paper: "Zhao et al., 2024",             desc: "Conv-transformer hybrid" },
    { name: "AttnSleep",         cat: "attention",     year: 2021, params: "1.2 M",  paper: "Eldele et al., 2021",           desc: "Attention sleep stager" },
    { name: "LMDA-Net",          cat: "attention",     year: 2024, params: "33 K",   paper: "Liu et al., 2024",              desc: "Lightweight multi-dim attention" },
    { name: "FBCNet",            cat: "filterbank",    year: 2021, params: "8 K",    paper: "Mane et al., 2021",             desc: "Filter-bank conv for MI" },
    { name: "FBLightConvNet",    cat: "filterbank",    year: 2023, params: "11 K",   paper: "Ma et al., 2023",               desc: "Lightweight filterbank ConvNet" },
    { name: "IFNet",             cat: "filterbank",    year: 2023, params: "30 K",   paper: "Wang et al., 2023",             desc: "Interactive frequency net" },
    { name: "FBMSNet",           cat: "filterbank",    year: 2022, params: "23 K",   paper: "Liu et al., 2022",              desc: "Multi-scale filter-bank net" },
    { name: "BENDR",             cat: "foundation",    year: 2021, params: "12 M",   paper: "Kostas et al., 2021",           desc: "Self-supervised EEG transformer" },
    { name: "Labram",            cat: "foundation",    year: 2024, params: "5.8 M",  paper: "Jiang et al., 2024",            desc: "Large brain-activity model" },
    { name: "BIOT",              cat: "foundation",    year: 2023, params: "3.2 M",  paper: "Yang et al., 2023",             desc: "Biosignal transformer" },
    { name: "EEGPT",             cat: "foundation",    year: 2024, params: "10 M",   paper: "Wang et al., 2024",             desc: "Pre-trained EEG transformer" },
    { name: "CBraMod",           cat: "foundation",    year: 2024, params: "9 M",    paper: "Wang et al., 2024",             desc: "Criss-cross brain modeling" },
    { name: "DeepSleepNet",      cat: "recurrent",     year: 2017, params: "21 M",   paper: "Supratak et al., 2017",         desc: "CNN + bidirectional LSTM" },
    { name: "ContraWR",          cat: "recurrent",     year: 2018, params: "1.4 M",  paper: "Lawhern (variant)",             desc: "Contrastive recurrent" },
    { name: "DGCNN",             cat: "graph",         year: 2020, params: "84 K",   paper: "Lun et al., 2020",              desc: "Dynamic graph convolution" },
    { name: "EEGProgress",       cat: "graph",         year: 2023, params: "120 K",  paper: "Chen et al., 2023",             desc: "Progressive graph net" },
    { name: "SyncNet",           cat: "channel",       year: 2017, params: "6 K",    paper: "Li et al., 2017",               desc: "Cross-channel sync filters" },
    { name: "EEGSym",            cat: "channel",       year: 2022, params: "92 K",   paper: "Pérez-Velasco et al., 2022",    desc: "Symmetric inter-hemisphere net" },
    { name: "SincShallowNet",    cat: "interpretable", year: 2020, params: "12 K",   paper: "Borra et al., 2020",            desc: "Sinc-conv shallow net" },
    { name: "Green",             cat: "interpretable", year: 2024, params: "70 K",   paper: "Paillard et al., 2024",         desc: "Geometry-aware EEG learner" },
    { name: "SPDNet",            cat: "spd",           year: 2017, params: "8 K",    paper: "Huang et al., 2017",            desc: "SPD manifold network" },
  ];

  var BD_CATEGORIES = injectedCategories() || [
    { id: "all",          label: "All",            color: "gray" },
    { id: "convolution",  label: "Convolution",    color: "" },
    { id: "attention",    label: "Attention",      color: "warm" },
    { id: "filterbank",   label: "Filterbank",     color: "green" },
    { id: "foundation",   label: "Foundation",     color: "purple" },
    { id: "recurrent",    label: "Recurrent",      color: "" },
    { id: "graph",        label: "Graph",          color: "green" },
    { id: "channel",      label: "Channel",        color: "warm" },
    { id: "interpretable",label: "Interpretable",  color: "" },
    { id: "spd",          label: "SPD",            color: "purple" },
  ];

  function bdInitDownloads(root) {
    var el = root.querySelector("#bd-downloads-num");
    if (!el) { return; }
    var doFetch = function () {
      // shields.io proxies pepy.tech and returns CORS-friendly JSON.
      fetch("https://img.shields.io/pepy/dt/braindecode.json")
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (j) {
          if (!j || !j.value) { return; }
          // value is "894k" / "1.2M" — uppercase the unit for visual consistency
          el.textContent = String(j.value).replace(/k$/, "K").replace(/m$/, "M");
        })
        .catch(function () { /* keep static fallback */ });
    };
    // Defer the cross-origin fetch until the facts strip is near the viewport
    // so bounce-traffic users don't pay for it.
    if (!("IntersectionObserver" in window)) { doFetch(); return; }
    var io = new IntersectionObserver(function (entries, obs) {
      if (entries.some(function (e) { return e.isIntersecting; })) {
        obs.disconnect();
        doFetch();
      }
    }, { rootMargin: "200px" });
    io.observe(el);
  }

  function bdInitInstall(root) {
    var btn = root.querySelector(".hero-install");
    if (!btn) { return; }
    btn.addEventListener("click", function () {
      if (!navigator.clipboard) { return; }
      navigator.clipboard.writeText("pip install braindecode").then(function () {
        btn.classList.add("is-copied");
        // CSS toggles between .icon-clipboard and .icon-check on .is-copied
        setTimeout(function () { btn.classList.remove("is-copied"); }, 1400);
      });
    });
  }

  function bdInitCodeStepper(root) {
    var steps = root.querySelectorAll(".code-step");
    var lines = root.querySelectorAll(".code-content > div[data-step]");
    if (!steps.length || !lines.length) { return; }
    var setActive = function (idx) {
      steps.forEach(function (s, i) { s.classList.toggle("active", i === idx); });
      lines.forEach(function (l) {
        l.classList.toggle("is-active", parseInt(l.dataset.step, 10) === idx);
      });
    };
    steps.forEach(function (s, i) {
      s.addEventListener("click", function () { setActive(i); });
    });
    setActive(0);
  }

  function bdInitZoo(root) {
    var toolbar = root.querySelector(".zoo-toolbar");
    var grid = root.querySelector(".zoo-grid");
    var foot = root.querySelector(".zoo-foot");
    if (!toolbar || !grid) { return; }

    var state = { cat: "all", q: "" };

    // Use Map (not plain object) for dynamic-key lookups so Codacy's
    // "Variable Assigned to Object Injection Sink" rule doesn't trip
    // on `counts[m.cat]` / `catById[id]` style access.
    var counts = new Map();
    counts.set("all", BD_MODELS.length);
    BD_MODELS.forEach(function (m) {
      counts.set(m.cat, (counts.get(m.cat) || 0) + 1);
    });

    // Cache category metadata by id so render() doesn't re-scan the array.
    var catById = new Map();
    BD_CATEGORIES.forEach(function (c) { catById.set(c.id, c); });
    function catLabel(id) { var c = catById.get(id); return c ? c.label : id; }
    function catColor(id) { var c = catById.get(id); return c ? c.color : ""; }

    // Predicates split out of filter() so each helper stays under
    // Codacy's cyclomatic-complexity-of-4 ceiling.
    function matchesCat(m) {
      return state.cat === "all" || m.cat === state.cat;
    }
    function matchesQuery(m, q) {
      if (!q) { return true; }
      var hay = (m.name + " " + m.paper + " " + m.desc).toLowerCase();
      return hay.includes(q);
    }

    // Helper: build one zoo card as a DOM tree (no innerHTML).
    function makeChild(tag, className, textNode) {
      var el = document.createElement(tag);
      if (className) { el.className = className; }
      if (textNode) { el.appendChild(document.createTextNode(textNode)); }
      return el;
    }
    function makeMetaSpan(text) {
      return makeChild("span", null, text);
    }
    function makeCard(m) {
      var card = makeChild("article", "zoo-card");
      var head = makeChild("div", "zoo-card-head");
      head.appendChild(makeChild("span", "zoo-card-name", m.name));
      head.appendChild(makeChild("span", "zoo-card-cat " + catColor(m.cat), catLabel(m.cat)));
      card.appendChild(head);
      if (m.paper) { card.appendChild(makeChild("span", "zoo-card-paper", m.paper)); }
      card.appendChild(makeChild("span", "zoo-card-desc", m.desc));
      var meta = makeChild("div", "zoo-card-meta");
      if (m.params) {
        var pSpan = document.createElement("span");
        var b = makeChild("b", null, String(m.params));
        pSpan.appendChild(b);
        pSpan.appendChild(document.createTextNode(" params"));
        meta.appendChild(pSpan);
      }
      if (m.year) {
        meta.appendChild(makeMetaSpan("·"));
        meta.appendChild(makeMetaSpan(String(m.year)));
      }
      card.appendChild(meta);
      return card;
    }

    function refreshChips() {
      toolbar.querySelectorAll(".zoo-chip").forEach(function (b) {
        b.classList.toggle("active", b.dataset.cat === state.cat);
      });
    }

    function refreshFooter(filteredLen) {
      if (!foot) { return; }
      var span = foot.querySelector(".zoo-foot-count");
      if (!span) { return; }
      span.textContent = filteredLen + " of " + BD_MODELS.length + " shown";
    }

    function paintEmpty() {
      grid.classList.add("is-empty");
      while (grid.firstChild) { grid.removeChild(grid.firstChild); }
      grid.appendChild(makeChild("div", "zoo-empty", "No models match — try another filter or query."));
    }

    function paintCards(filtered) {
      grid.classList.remove("is-empty");
      var frag = document.createDocumentFragment();
      filtered.forEach(function (m) { frag.appendChild(makeCard(m)); });
      while (grid.firstChild) { grid.removeChild(grid.firstChild); }
      grid.appendChild(frag);
    }

    function render() {
      refreshChips();
      var q = state.q.toLowerCase();
      var filtered = BD_MODELS.filter(function (m) {
        return matchesCat(m) && matchesQuery(m, q);
      });
      if (filtered.length === 0) { paintEmpty(); }
      else { paintCards(filtered); }
      refreshFooter(filtered.length);
    }

    // Build chips (now that render() is defined above us so the click
    // handler doesn't reference an undeclared name).
    var searchEl = toolbar.querySelector(".zoo-search");
    BD_CATEGORIES.forEach(function (c) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "zoo-chip" + (c.id === "all" ? " active" : "");
      btn.dataset.cat = c.id;
      // Build the chip via DOM nodes (textContent / appendChild) instead
      // of innerHTML — Codacy's "Unsafe assignment to innerHTML" rule
      // doesn't trip and this is faster than parsing an HTML string.
      btn.appendChild(document.createTextNode(c.label + " "));
      var countSpan = document.createElement("span");
      countSpan.className = "chip-count";
      countSpan.textContent = String(counts.get(c.id) || 0);
      btn.appendChild(countSpan);
      btn.addEventListener("click", function () { state.cat = c.id; render(); });
      toolbar.insertBefore(btn, searchEl);
    });

    // Wire search
    var input = root.querySelector(".zoo-search input");
    if (input) {
      input.addEventListener("input", function (e) { state.q = e.target.value; render(); });
    }

    render();
  }

  function bdInit() {
    var root = document.querySelector(".bd-landing");
    if (!root) { return; }
    [bdInitInstall, bdInitCodeStepper, bdInitZoo, bdInitDownloads].forEach(function (fn) {
      try {
        fn(root);
      } catch (e) {
        var w = window.console || { warn() {} };
        w.warn("[bd-landing] " + (fn.name || "init") + " failed:", e);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bdInit);
  } else {
    bdInit();
  }
}());
