/* One-click citation export for header banner, cite.html, and landing footer. */

const ZENODO_BIBTEX = [
  "@software{braindecode,",
  "  author = {Aristimunha, Bruno and",
  "            Guetschel, Pierre and",
  "            Wimpff, Martin and",
  "            Gemein, Lukas and",
  "            Rommel, Cedric and",
  "            Banville, Hubert and",
  "            Sliwowski, Maciej and",
  "            Wilson, Daniel and",
  "            Brandt, Simon and",
  "            Gnassounou, Th{\\'e}o and",
  "            Paillard, Joseph and",
  "            Srivastava, Aman and",
  "            {Junqueira Lopes}, Bruna and",
  "            Sedlar, Sara and",
  "            Tayal, Sarthak and",
  "            Truong, Young and",
  "            Gadonneix, Julien and",
  "            Mounir, Adam and",
  "            Zhang, Zhibo and",
  "            Soloviev, Anton and",
  "            Li, Qing and",
  "            Rathi, Raghav and",
  "            Shah, Vandit and",
  "            Arthur and",
  "            Jain, Mahir and",
  "            Kowshik, Bhargav and",
  "            Li, Yiheng and",
  "            Siddique, Fashad Ahmed and",
  "            Patel, Aayushya and",
  "            Moreau, Thomas and",
  "            Chevallier, Sylvain and",
  "            Gramfort, Alexandre and",
  "            Schirrmeister, Robin Tibor},",
  "  title = {Braindecode: toolbox for decoding raw electrophysiological brain data with deep learning models},",
  "  year = {2026},",
  "  publisher = {Zenodo},",
  "  howpublished = {Zenodo},",
  "  url = {https://github.com/braindecode/braindecode},",
  "  doi = {10.5281/zenodo.8214376},",
  "  license = {BSD-3-Clause}",
  "}",
].join("\n");

const HBM_BIBTEX = [
  "@article{HBM:HBM23730,",
  "  author = {Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and",
  "            Fiederer, Lukas Dominique Josef and Glasstetter, Martin and",
  "            Eggensperger, Katharina and Tangermann, Michael and",
  "            Hutter, Frank and Burgard, Wolfram and Ball, Tonio},",
  "  title = {Deep learning with convolutional neural networks for {EEG} decoding and visualization},",
  "  journal = {Human Brain Mapping},",
  "  issn = {1097-0193},",
  "  url = {https://doi.org/10.1002/hbm.23730},",
  "  doi = {10.1002/hbm.23730},",
  "  month = {aug},",
  "  year = {2017}",
  "}",
].join("\n");

const ZENODO_APA =
  "Aristimunha, B., Guetschel, P., Wimpff, M., Gemein, L., Rommel, C., Banville, H., Sliwowski, M., Wilson, D., Brandt, S., Gnassounou, T., Paillard, J., Srivastava, A., Junqueira Lopes, B., Sedlar, S., Tayal, S., Truong, Y., Gadonneix, J., Mounir, A., Zhang, Z., Soloviev, A., Li, Q., Rathi, R., Shah, V., Arthur, Jain, M., Kowshik, B., Li, Y., Siddique, F. A., Patel, A., Moreau, T., Chevallier, S., Gramfort, A., & Schirrmeister, R. T. (2026). Braindecode: toolbox for decoding raw electrophysiological brain data with deep learning models [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.8214376";

const HBM_APA =
  "Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping. https://doi.org/10.1002/hbm.23730";

const ZENODO_RIS = [
  "TY  - COMP",
  "TI  - Braindecode: toolbox for decoding raw electrophysiological brain data with deep learning models",
  "AU  - Aristimunha, Bruno",
  "AU  - Guetschel, Pierre",
  "AU  - Wimpff, Martin",
  "AU  - Gemein, Lukas",
  "AU  - Rommel, Cedric",
  "AU  - Banville, Hubert",
  "AU  - Sliwowski, Maciej",
  "AU  - Wilson, Daniel",
  "AU  - Brandt, Simon",
  "AU  - Gnassounou, Théo",
  "AU  - Paillard, Joseph",
  "AU  - Srivastava, Aman",
  "AU  - Junqueira Lopes, Bruna",
  "AU  - Sedlar, Sara",
  "AU  - Tayal, Sarthak",
  "AU  - Truong, Young",
  "AU  - Gadonneix, Julien",
  "AU  - Mounir, Adam",
  "AU  - Zhang, Zhibo",
  "AU  - Soloviev, Anton",
  "AU  - Li, Qing",
  "AU  - Rathi, Raghav",
  "AU  - Shah, Vandit",
  "AU  - Arthur",
  "AU  - Jain, Mahir",
  "AU  - Kowshik, Bhargav",
  "AU  - Li, Yiheng",
  "AU  - Siddique, Fashad Ahmed",
  "AU  - Patel, Aayushya",
  "AU  - Moreau, Thomas",
  "AU  - Chevallier, Sylvain",
  "AU  - Gramfort, Alexandre",
  "AU  - Schirrmeister, Robin Tibor",
  "PY  - 2026",
  "PB  - Zenodo",
  "DO  - 10.5281/zenodo.8214376",
  "UR  - https://doi.org/10.5281/zenodo.8214376",
  "ER  - ",
].join("\n");

const CITE_PAYLOADS = new Map([
  ["zenodo-bibtex", ZENODO_BIBTEX],
  ["zenodo-apa", ZENODO_APA],
  ["hbm-bibtex", HBM_BIBTEX],
  ["hbm-apa", HBM_APA],
  ["both-bibtex", ZENODO_BIBTEX + "\n\n" + HBM_BIBTEX],
  ["zenodo-ris", ZENODO_RIS],
]);

function fallbackCopy(text) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.top = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand("copy");
  document.body.removeChild(ta);
  return ok;
}

function writeCiteText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  return fallbackCopy(text) ? Promise.resolve() : Promise.reject(new Error("copy failed"));
}

function markCiteCopied(btn) {
  if (!btn.dataset.bdOrigLabel) {
    btn.dataset.bdOrigLabel = btn.textContent;
  }
  clearTimeout(btn._bdCopyTimer);
  btn.setAttribute("data-copied", "true");
  btn.textContent = "\u2713 Copied!";
  const live = document.querySelector(".bd-cite-live");
  if (live) {
    live.textContent = "Copied citation to clipboard.";
  }
  btn._bdCopyTimer = setTimeout(() => {
    btn.removeAttribute("data-copied");
    btn.textContent = btn.dataset.bdOrigLabel;
  }, 1800);
}

function handleCiteClick(event) {
  const btn = event.target?.closest?.("[data-bd-cite-copy]");
  if (!btn) {
    return;
  }
  const text = CITE_PAYLOADS.get(btn.getAttribute("data-bd-cite-copy"));
  if (!text) {
    return;
  }
  event.preventDefault();
  writeCiteText(text)
    .then(() => {
      markCiteCopied(btn);
    })
    .catch(() => null);
}

document.addEventListener("click", handleCiteClick);
