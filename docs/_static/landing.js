const copyTimers = new WeakMap();
let lastCopyMessage = "";

async function writeClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    return Boolean(err && false);
  }
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

async function handleCopyClick(event) {
  const btn = event.target.closest("[data-copy]");
  if (!btn) {
    return;
  }
  const parent = btn.parentElement;
  const status = parent ? parent.querySelector(".bd-copy-status") : null;
  clearTimeout(copyTimers.get(btn));
  const ok = await writeClipboard(btn.dataset.copy);
  btn.toggleAttribute("data-copied", ok);
  announceCopy(status, ok);
  const timer = setTimeout(() => {
    clearCopyFeedback(btn, status);
  }, 1800);
  copyTimers.set(btn, timer);
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

function isZooExpanded(query, mod, cat, moreBox) {
  if (query || mod !== "all") {
    return true;
  }
  if (cat !== "all") {
    return true;
  }
  return Boolean(moreBox && moreBox.checked);
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
    const raw = searchInput ? searchInput.value : "";
    const query = raw.trim().toLowerCase();
    const mod = readCheckedValue(zoo, "bd-zoo-mod");
    const cat = readCheckedValue(zoo, "bd-zoo-cat");
    zoo.toggleAttribute("data-searching", Boolean(query));
    let matched = 0;
    for (const card of cards) {
      card.hidden = Boolean(query) && !card.dataset.search.includes(query);
      if (cardMatchesFilters(card, mod, cat)) {
        matched += 1;
      }
    }
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
    .catch((err) => Boolean(err && false));
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
