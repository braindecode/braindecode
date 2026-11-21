const copyTimers = new WeakMap();
let lastCopyMessage = "";

function writeClipboard(text) {
  if (!navigator.clipboard) {
    return Promise.resolve(false);
  }
  return navigator.clipboard.writeText(text).then(
    () => true,
    () => false
  );
}

function announceCopy(status, ok) {
  if (!status) {
    return;
  }
  let msg = "Copy failed";
  if (ok) {
    msg = "Copied to clipboard";
  }
  if (msg === lastCopyMessage) {
    msg += "\u00A0";
  }
  lastCopyMessage = msg;
  status.textContent = msg;
}

function clearCopyFeedback(btn, status) {
  btn.removeAttribute("data-copied");
  if (status) {
    status.textContent = "";
  }
}

function findStatusNode(btn) {
  const parent = btn.parentElement;
  if (!parent) {
    return null;
  }
  return parent.querySelector(".bd-copy-status");
}

function handleCopyClick(event) {
  const btn = event.target.closest("[data-copy]");
  if (!btn) {
    return;
  }
  const status = findStatusNode(btn);
  clearTimeout(copyTimers.get(btn));
  writeClipboard(btn.dataset.copy).then((ok) => {
    btn.toggleAttribute("data-copied", ok);
    announceCopy(status, ok);
    const timer = setTimeout(() => {
      clearCopyFeedback(btn, status);
    }, 1800);
    copyTimers.set(btn, timer);
  });
}

function readCheckedValue(zoo, name) {
  const el = zoo.querySelector("input[name=\"" + name + "\"]:checked");
  if (el) {
    return el.value;
  }
  return "all";
}

function matchesToken(listStr, selected) {
  if (selected === "all") {
    return true;
  }
  return listStr.split(" ").includes(selected);
}

function cardMatchesFilters(card, mod, cat) {
  if (card.hidden) {
    return false;
  }
  if (!matchesToken(card.dataset.mod, mod)) {
    return false;
  }
  return matchesToken(card.dataset.cats, cat);
}

function hasActiveFilter(query, mod, cat) {
  if (query) {
    return true;
  }
  return mod !== "all" || cat !== "all";
}

function isMoreChecked(moreBox) {
  if (!moreBox) {
    return false;
  }
  return Boolean(moreBox.checked);
}

function isZooExpanded(query, mod, cat, moreBox) {
  if (hasActiveFilter(query, mod, cat)) {
    return true;
  }
  return isMoreChecked(moreBox);
}

function updateZooCount(countOut, expanded, matched, total) {
  if (!countOut) {
    return;
  }
  if (expanded) {
    countOut.textContent = "Showing " + matched + " of " + total + " architectures";
    return;
  }
  countOut.textContent = "Showing 12 featured of " + total + " architectures";
}

function readSearchQuery(searchInput) {
  if (!searchInput) {
    return "";
  }
  return searchInput.value.trim().toLowerCase();
}

function applyCardSearch(card, query) {
  if (!query) {
    card.hidden = false;
    return;
  }
  card.hidden = !card.dataset.search.includes(query);
}

function countVisibleCards(cards, query, mod, cat) {
  let matched = 0;
  for (const card of cards) {
    applyCardSearch(card, query);
    if (cardMatchesFilters(card, mod, cat)) {
      matched += 1;
    }
  }
  return matched;
}

function isEditableTarget(target) {
  if (!target || !target.matches) {
    return false;
  }
  return target.matches("input, textarea, select, [contenteditable]");
}

function hasKeyModifier(event) {
  if (event.metaKey || event.ctrlKey) {
    return true;
  }
  return event.altKey;
}

function bindSlashShortcut(searchInput) {
  if (!searchInput) {
    return;
  }
  document.addEventListener("keydown", (event) => {
    if (event.key !== "/" || hasKeyModifier(event)) {
      return;
    }
    if (!isEditableTarget(event.target)) {
      event.preventDefault();
      searchInput.focus();
    }
  });
}

function initZoo(root) {
  const zoo = root.querySelector(".bd-zoo");
  if (!zoo) {
    return;
  }
  const cards = [...zoo.querySelectorAll(".zoo-card")];
  const searchInput = zoo.querySelector("#bd-zoo-q");
  const moreBox = zoo.querySelector("#bd-zoo-more");
  const countOut = zoo.querySelector(".zoo-foot-count");
  const syncZoo = () => {
    const query = readSearchQuery(searchInput);
    const mod = readCheckedValue(zoo, "bd-zoo-mod");
    const cat = readCheckedValue(zoo, "bd-zoo-cat");
    const matched = countVisibleCards(cards, query, mod, cat);
    zoo.toggleAttribute("data-searching", Boolean(query));
    zoo.toggleAttribute("data-empty", matched === 0);
    updateZooCount(countOut, isZooExpanded(query, mod, cat, moreBox), matched, cards.length);
  };
  if (searchInput) {
    searchInput.addEventListener("input", syncZoo);
  }
  zoo.addEventListener("change", syncZoo);
  bindSlashShortcut(searchInput);
}

function setActiveVenueDot(dots, activeIdx) {
  dots.forEach((dot, idx) => {
    dot.toggleAttribute("data-active", idx === activeIdx);
    dot.setAttribute("aria-selected", idx === activeIdx ? "true" : "false");
  });
}

function scrollTrackToIndex(track, cards, idx) {
  const target = cards[idx];
  if (!target) {
    return;
  }
  const offset = target.offsetLeft - track.offsetLeft - 32;
  track.scrollTo({ left: Math.max(0, offset), behavior: "smooth" });
}

function computeActiveCardIndex(track, cards) {
  const scrollEdge = track.scrollLeft + 96;
  let best = 0;
  cards.forEach((card, idx) => {
    if (card.offsetLeft - track.offsetLeft <= scrollEdge) {
      best = idx;
    }
  });
  return best;
}

function bindCarouselButtons(wrap, track, cards, dots) {
  const prevBtn = wrap.querySelector("[data-venue-prev]");
  const nextBtn = wrap.querySelector("[data-venue-next]");
  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      const next = Math.max(0, computeActiveCardIndex(track, cards) - 1);
      scrollTrackToIndex(track, cards, next);
    });
  }
  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      const next = Math.min(cards.length - 1, computeActiveCardIndex(track, cards) + 1);
      scrollTrackToIndex(track, cards, next);
    });
  }
  dots.forEach((dot, idx) => {
    dot.addEventListener("click", () => {
      scrollTrackToIndex(track, cards, idx);
    });
  });
}

function initVenueCarousel(root) {
  const wrap = root.querySelector(".venue-showcase");
  if (!wrap) {
    return;
  }
  const track = wrap.querySelector(".venue-track");
  if (!track) {
    return;
  }
  const cards = [...track.querySelectorAll(".venue-card")];
  const dots = [...wrap.querySelectorAll(".venue-dot")];
  bindCarouselButtons(wrap, track, cards, dots);
  track.addEventListener(
    "scroll",
    () => {
      setActiveVenueDot(dots, computeActiveCardIndex(track, cards));
    },
    { passive: true }
  );
}

function applyDownloadBadge(dlEl, payload) {
  const raw = payload?.value;
  if (!raw) {
    return;
  }
  const clean = String(raw).replace(/k$/, "K").replace(/m$/, "M");
  dlEl.textContent = clean.replace(/\+?$/, "+");
}

function fetchDownloads(dlEl) {
  fetch("https://img.shields.io/pepy/dt/braindecode.json", { priority: "low" })
    .then((resp) => (resp.ok ? resp.json() : null))
    .then((payload) => {
      applyDownloadBadge(dlEl, payload);
    })
    .catch(() => null);
}

function initDownloads(root) {
  const dlEl = root.querySelector("#bd-downloads-num");
  if (!dlEl) {
    return;
  }
  const io = new IntersectionObserver(
    (entries, obs) => {
      if (!entries.some((entry) => entry.isIntersecting)) {
        return;
      }
      obs.disconnect();
      fetchDownloads(dlEl);
    },
    { rootMargin: "200px" }
  );
  io.observe(dlEl);
}

function placeApiPop(pop, anchor) {
  const rect = anchor.getBoundingClientRect();
  const popW = Math.min(440, window.innerWidth - 32);
  const left = Math.min(Math.max(16, rect.left), window.innerWidth - popW - 16);
  const below = rect.bottom + 10;
  const top = below + 132 > window.innerHeight ? Math.max(16, rect.top - 118) : below;
  pop.style.left = left + "px";
  pop.style.top = top + "px";
}

function showApiPop(pop, titleEl, codeEl, target) {
  titleEl.textContent = target.getAttribute("data-api-head");
  codeEl.textContent = target.getAttribute("data-api-code");
  pop.hidden = false;
  placeApiPop(pop, target);
}

function handleZooEnter(pop, titleEl, codeEl, event) {
  const item = event.target?.closest?.("[data-api-code]");
  if (!item) {
    return;
  }
  showApiPop(pop, titleEl, codeEl, item);
}

function handleZooLeave(pop, event) {
  const fromItem = event.target?.closest?.("[data-api-code]");
  if (!fromItem) {
    return;
  }
  if (!fromItem.contains(event.relatedTarget)) {
    pop.hidden = true;
  }
}

function initZooApiPreview(root) {
  const zoo = root.querySelector(".bd-zoo");
  const pop = root.querySelector("#bd-zoo-pop");
  if (!zoo || !pop) {
    return;
  }
  const titleEl = pop.querySelector(".zoo-api-pop-title");
  const codeEl = pop.querySelector(".zoo-api-pop-code");
  const onEnter = (event) => handleZooEnter(pop, titleEl, codeEl, event);
  const hidePop = () => {
    pop.hidden = true;
  };
  zoo.addEventListener("pointerover", onEnter);
  zoo.addEventListener("pointerout", (event) => handleZooLeave(pop, event));
  zoo.addEventListener("focusin", onEnter);
  zoo.addEventListener("focusout", hidePop);
  window.addEventListener("scroll", hidePop, { passive: true });
}

function runCountUp(nodes, target) {
  const start = performance.now();
  const duration = 1050;
  const step = (now) => {
    const p = Math.min(1, (now - start) / duration);
    const val = Math.max(1, Math.round((1 - Math.pow(1 - p, 3)) * target));
    nodes.forEach((el) => {
      el.textContent = String(val);
    });
    if (p < 1) {
      requestAnimationFrame(step);
    }
  };
  requestAnimationFrame(step);
}

function prefersReducedMotion() {
  if (!window.matchMedia) {
    return false;
  }
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function initCountUp(root) {
  const nodes = [...root.querySelectorAll("[data-count-to]")];
  if (!nodes.length || prefersReducedMotion()) {
    return;
  }
  const target = Number(nodes[0].dataset.countTo);
  const io = new IntersectionObserver(
    (entries, obs) => {
      if (!entries.some((entry) => entry.isIntersecting)) {
        return;
      }
      obs.disconnect();
      runCountUp(nodes, target);
    },
    { threshold: 0.25 }
  );
  io.observe(nodes[0]);
}

function setTextIfPresent(el, text) {
  if (el && text) {
    el.textContent = text;
  }
}

function applyActiveCap(stage, cards, modEl, exprEl, activeIdx) {
  const card = cards[activeIdx];
  if (!card) {
    return;
  }
  stage.dataset.activeCap = String(activeIdx);
  cards.forEach((item, idx) => {
    item.toggleAttribute("data-cap-active", idx === activeIdx);
  });
  setTextIfPresent(modEl, card.dataset.capMod);
  setTextIfPresent(exprEl, card.dataset.capSig);
}

function bindCapableHover(stage, cards, modEl, exprEl, setLocked) {
  const inputs = stage.querySelector(".capable-inputs");
  if (!inputs) {
    return;
  }
  cards.forEach((card, idx) => {
    const lockTo = () => {
      setLocked(true, idx);
      applyActiveCap(stage, cards, modEl, exprEl, idx);
    };
    card.addEventListener("pointerenter", lockTo);
    card.addEventListener("focusin", lockTo);
  });
  inputs.addEventListener("pointerleave", () => {
    setLocked(false, -1);
  });
  inputs.addEventListener("focusout", (event) => {
    if (!inputs.contains(event.relatedTarget)) {
      setLocked(false, -1);
    }
  });
}

function observeCapableCycle(stage, stepFn) {
  let timer = 0;
  const io = new IntersectionObserver(
    (entries) => {
      const visible = entries.some((entry) => entry.isIntersecting);
      clearInterval(timer);
      timer = visible ? setInterval(stepFn, 2400) : 0;
    },
    { rootMargin: "120px" }
  );
  io.observe(stage);
}

function initCapableStage(root) {
  const stage = root.querySelector(".capable-stage");
  if (!stage) {
    return;
  }
  const cards = [...stage.querySelectorAll(".cap-card")];
  if (!cards.length) {
    return;
  }
  const modEl = stage.querySelector("[data-cap-live-mod]");
  const exprEl = stage.querySelector("[data-cap-live-expr]");
  let activeIdx = 0;
  let locked = false;
  const setLocked = (nextLocked, nextIdx) => {
    locked = nextLocked;
    stage.toggleAttribute("data-cap-locked", nextLocked);
    if (nextIdx >= 0) {
      activeIdx = nextIdx;
    }
  };
  bindCapableHover(stage, cards, modEl, exprEl, setLocked);
  if (prefersReducedMotion()) {
    return;
  }
  observeCapableCycle(stage, () => {
    if (!locked) {
      activeIdx = (activeIdx + 1) % cards.length;
      applyActiveCap(stage, cards, modEl, exprEl, activeIdx);
    }
  });
}

const landingRoot = document.querySelector(".bd-landing");
if (landingRoot) {
  landingRoot.addEventListener("click", handleCopyClick);
  initZoo(landingRoot);
  initZooApiPreview(landingRoot);
  initCountUp(landingRoot);
  initCapableStage(landingRoot);
  initVenueCarousel(landingRoot);
  initDownloads(landingRoot);
}
