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

function applyDownloadBadge(dlEl, payload) {
  if (!payload || !payload.value) {
    return;
  }
  dlEl.textContent = String(payload.value).replace(/k$/, "K").replace(/m$/, "M");
}

function fetchDownloads(dlEl) {
  fetch("https://img.shields.io/pepy/dt/braindecode.json", { priority: "low" })
    .then((resp) => {
      if (resp.ok) {
        return resp.json();
      }
      return null;
    })
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

const landingRoot = document.querySelector(".bd-landing");
if (landingRoot) {
  landingRoot.addEventListener("click", handleCopyClick);
  initZoo(landingRoot);
  initDownloads(landingRoot);
}
