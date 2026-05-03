:html_theme.sidebar_secondary.remove: true

.. meta::
    :description: Braindecode is an open-source PyTorch toolbox for decoding raw EEG, ECoG and MEG with deep learning. Datasets, preprocessing, augmentations, and 65+ published model architectures.
    :keywords: braindecode, EEG decoding, ECoG, MEG, deep learning, PyTorch, MNE-Python, MOABB, EEGDash

..
    The page title is required so Sphinx can wire next/prev navigation
    and emit a sensible <title> + h1 for SEO. We hide it visually only
    on this page; it's still in the DOM for crawlers.

.. raw:: html

    <style type="text/css">section[id^="braindecode-decode"] > h1 {display:none;}</style>
    <div class="bd-landing">

###############################################################
 Braindecode — Decode raw EEG, ECoG and MEG with deep learning
###############################################################

.. ====================================================================

.. HERO

.. ====================================================================

.. raw:: html

    <section class="hero">
      <div class="container hero-grid">
        <div class="hero-text">
          <div class="hero-eyebrow">
            <span class="dot"></span>
            <span>EEG · ECoG · MEG · EMG · iEEG &nbsp;·&nbsp; Open source since 2017</span>
          </div>
          <h1 class="hero-title">
            Decode raw <em>brain signals</em> with <span class="highlight">deep learning.</span>
          </h1>
          <p class="hero-lede">
            A PyTorch-native toolbox for end-to-end neural decoding.
            Load a dataset, pick a published model, train. Same scikit-learn
            API you already know.
          </p>
          <div class="hero-cta-row">
            <a href="install/install.html" class="btn btn-primary">Get started <span class="btn-arrow">→</span></a>
            <a href="auto_examples/index.html" class="btn">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 4 20 12 6 20 6 4"/></svg>
              Tutorials
            </a>
            <a href="api.html" class="btn">API reference</a>
            <a href="https://github.com/braindecode/braindecode" target="_blank" rel="noopener" class="btn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-4.3 1.4-4.3-2.5-6-3m12 5v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.2 4.2 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"/></svg>
              GitHub
            </a>
          </div>
          <button type="button" class="hero-install" aria-label="Copy install command">
            <span class="prompt">$</span>
            <span>pip install braindecode</span>
            <span class="copy-icon" aria-hidden="true">
              <svg class="icon-clipboard" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              <svg class="icon-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
            </span>
          </button>
          <p class="hero-tagline-emphasis">
            For neuroscientists who want to work with deep learning, and deep
            learning researchers who want to work with neurophysiological data.
          </p>

          <div class="hero-meta">
            <span class="meta-item"><strong>v1.5</strong> · BSD-3-Clause</span>
            <span class="meta-item">Python <strong>3.11+</strong></span>
            <span class="meta-item">Works with <strong>MNE-Python</strong></span>
          </div>

          <div class="hero-side-row">
            <div><span class="num">2017</span><span class="lbl">First release</span></div>
            <div><span class="num">80<sup>+</sup></span><span class="lbl">Contributors</span></div>
            <div><span class="num">PyTorch</span><span class="lbl">+ scikit-learn · + MNE</span></div>
          </div>
        </div>

        <div class="hero-art" aria-hidden="true">
          <svg class="hero-art-svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 60 600 280" preserveAspectRatio="xMidYMid meet">
            <defs>
              <filter id="bd-glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="b"/>
                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>

            <image xlink:href="_static/brain_only.png" x="6" y="78" width="138" height="160" preserveAspectRatio="xMidYMid meet" class="hero-brain-img"/>

            <g class="bd-eeg" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round">
              <path class="bd-eeg-line ch1" d="M130 116 q 6 -8 12 0 t 12 0 t 12 0 t 12 0 q 4 -6 8 0 q 4 5 8 0 q 4 -7 8 0 q 6 -8 12 0 t 12 0 t 12 0 t 12 0 q 6 -7 12 0 t 12 0 t 12 0 t 12 0 q 6 -6 12 0 t 12 0 t 12 0 L 450 116"/>
              <path class="bd-eeg-line ch2" d="M130 144 q 12 -12 24 0 t 24 0 t 24 0 q 12 -8 24 0 t 24 0 t 24 0 q 12 -10 24 0 t 24 0 L 450 144"/>
              <path class="bd-eeg-line ch3" d="M130 172 q 8 -14 16 0 t 16 0 t 16 0 t 16 0 q 8 -10 16 0 t 16 0 t 16 0 t 16 0 q 8 -12 16 0 t 16 0 t 16 0 L 450 172"/>
              <path class="bd-eeg-line ch4" d="M130 200 q 4 -14 8 0 t 8 0 t 8 0 t 8 0 t 8 0 t 8 0 t 8 0 t 8 0 q 4 -10 8 0 t 8 0 t 8 0 t 8 0 t 8 0 t 8 0 t 8 0 q 4 -12 8 0 t 8 0 t 8 0 t 8 0 L 450 200"/>
            </g>

            <rect class="bd-window" x="250" y="100" width="44" height="124" rx="3" fill="#3a6c97" fill-opacity=".09" stroke="#3a6c97" stroke-opacity=".55" stroke-dasharray="4 3" stroke-width="1.2"/>

            <g class="bd-net">
              <g class="bd-edges-1" stroke="#3a6c97" stroke-width="1" stroke-opacity=".55" fill="none">
                <line x1="462" y1="116" x2="512" y2="130"/><line x1="462" y1="116" x2="512" y2="158"/><line x1="462" y1="116" x2="512" y2="186"/>
                <line x1="462" y1="144" x2="512" y2="130"/><line x1="462" y1="144" x2="512" y2="158"/><line x1="462" y1="144" x2="512" y2="186"/>
                <line x1="462" y1="172" x2="512" y2="130"/><line x1="462" y1="172" x2="512" y2="158"/><line x1="462" y1="172" x2="512" y2="186"/>
                <line x1="462" y1="200" x2="512" y2="130"/><line x1="462" y1="200" x2="512" y2="158"/><line x1="462" y1="200" x2="512" y2="186"/>
              </g>
              <g class="bd-edges-2" stroke="#3a6c97" stroke-width="1.2" stroke-opacity=".7" fill="none">
                <line x1="512" y1="130" x2="556" y2="158"/><line x1="512" y1="158" x2="556" y2="158"/><line x1="512" y1="186" x2="556" y2="158"/>
              </g>
              <g class="bd-nodes-in">
                <circle class="n n-c1" cx="462" cy="116" r="7"/>
                <circle class="n n-c2" cx="462" cy="144" r="7"/>
                <circle class="n n-c3" cx="462" cy="172" r="7"/>
                <circle class="n n-c4" cx="462" cy="200" r="7"/>
              </g>
              <g class="bd-nodes-h" fill="#3a6c97">
                <circle cx="512" cy="130" r="7"/>
                <circle cx="512" cy="158" r="7"/>
                <circle cx="512" cy="186" r="7"/>
              </g>
              <circle class="bd-readout" cx="556" cy="158" r="10" fill="#b46a2c" filter="url(#bd-glow)"/>
              <text class="bd-readout-label" x="556" y="192" text-anchor="middle">right hand</text>
            </g>
          </svg>
        </div>
      </div>
    </section>

.. ====================================================================

.. INSTITUTION STRIP

.. ====================================================================

.. raw:: html

    <section class="logo-strip">
      <div class="container logo-strip-inner">
        <span class="logo-strip-label">Built &amp; maintained at</span>
        <div class="logo-strip-logos">
          <img src="_static/institution_logos/ucsd.png" alt="UC San Diego" style="height:28px"/>
          <img src="_static/institution_logos/unifreiburg.png" alt="Uni Freiburg" style="height:38px"/>
          <img src="_static/institution_logos/inria.png" alt="Inria" style="height:22px"/>
          <img src="_static/institution_logos/donders.svg" alt="Donders Institute" style="height:26px"/>
        </div>
        <span class="logo-strip-label" style="text-align:right">+ 80 community contributors</span>
      </div>
    </section>

.. ====================================================================

.. FACTS STRIP

.. ====================================================================

.. raw:: html

    <section class="bd-facts">
      <div class="container bd-facts-inner">
        <span class="bd-facts-label">By the numbers</span>
        <div class="bd-facts-row">
          <a href="models/models.html"><b>65+</b> models</a>
          <a href="api.html#augmentation"><b>20+</b> augmentations</a>
          <a href="https://moabb.neurotechx.com/" target="_blank" rel="noopener"><b>150+</b> datasets <span style="color:var(--bd-muted)">via MOABB</span></a>
          <a href="https://eegdash.org/" target="_blank" rel="noopener"><b>700+</b> datasets <span style="color:var(--bd-muted)">via EEGDash</span></a>
          <a href="auto_examples/index.html"><b>35+</b> tutorials</a>
          <a href="https://github.com/braindecode/braindecode/graphs/contributors" target="_blank" rel="noopener"><b>80+</b> contributors</a>
          <a href="https://www.pepy.tech/projects/braindecode" target="_blank" rel="noopener" class="bd-live-stat" id="bd-downloads-stat" title="Live total PyPI downloads via pepy.tech">
            <span class="bd-live-dot" aria-hidden="true"></span>
            <b id="bd-downloads-num">894K</b> downloads
          </a>
          <a href="whats_new.html"><b>9 yrs</b> of releases</a>
        </div>
        <span class="bd-facts-meta">open source since 2017</span>
      </div>
    </section>

.. ====================================================================

.. METRICS

.. ====================================================================

.. raw:: html

    <section class="section section-tight">
      <div class="container">
        <div class="metrics">
          <div class="metric-cell">
            <div class="metric-num">65<span class="plus">+</span></div>
            <div class="metric-label">Published model architectures</div>
          </div>
          <div class="metric-cell">
            <div class="metric-num">20<span class="plus">+</span></div>
            <div class="metric-label">EEG-specific augmentations</div>
          </div>
          <a class="metric-cell" href="https://moabb.neurotechx.com/" target="_blank" rel="noopener" style="text-decoration:none;color:inherit">
            <div class="metric-num">150<span class="plus">+</span></div>
            <div class="metric-label">Public datasets via <strong>MOABB</strong></div>
          </a>
          <a class="metric-cell" href="https://eegdash.org/dataset_summary.html" target="_blank" rel="noopener" style="text-decoration:none;color:inherit">
            <div class="metric-num">700<span class="plus">+</span></div>
            <div class="metric-label">BIDS datasets via <strong>EEGDash</strong></div>
          </a>
        </div>
      </div>
    </section>

.. ====================================================================

.. FEATURES

.. ====================================================================

.. raw:: html

    <section class="section">
      <div class="container">
        <div class="section-head">
          <div class="section-eyebrow">Why braindecode</div>
          <h2 class="section-title">Everything you need to go <em>from signal to prediction.</em></h2>
          <p class="section-sub">Built to <strong>plug into the EEG ecosystem you already use</strong>: every <a href="https://moabb.neurotechx.com/" target="_blank" rel="noopener">MOABB</a> dataset, every <a href="https://mne.tools/" target="_blank" rel="noopener">MNE-Python</a> preprocessing function, every <a href="https://scikit-learn.org/" target="_blank" rel="noopener">scikit-learn</a> training loop, plus 700+ BIDS datasets via <a href="https://eegdash.org/" target="_blank" rel="noopener">EEGDash</a>. One library, no lock-in.</p>
        </div>
        <div class="feature-grid">
          <a href="models/models.html" class="feature-card">
            <span class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0 4 4 2 0"/></svg></span>
            <h3>Decode raw electrophysiology</h3>
            <p>End-to-end models go straight from raw EEG/ECoG/MEG to predictions. No hand-crafted features required.</p>
            <span class="feature-link">Models <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></span>
          </a>
          <a href="api.html" class="feature-card">
            <span class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v6m0 8v6M4.93 4.93l4.24 4.24m5.66 5.66 4.24 4.24M2 12h6m8 0h6M4.93 19.07l4.24-4.24m5.66-5.66 4.24-4.24"/></svg></span>
            <h3>PyTorch-native, sklearn-friendly</h3>
            <p>Built on PyTorch and skorch with a Lightning-friendly API. Drop into any training loop you already know.</p>
            <span class="feature-link">API reference <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></span>
          </a>
          <a href="api.html#datasets" class="feature-card">
            <span class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/></svg></span>
            <h3>Every MOABB dataset, drop-in</h3>
            <p>All <b>150+ MOABB datasets</b> work out of the box, with <code>MOABBDataset(...)</code>. Plus TUH, HBN, Sleep Physionet, BCI Competition IV, BIDS, and 700+ datasets via sibling project EEGDash.</p>
            <span class="feature-link">Datasets <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></span>
          </a>
          <div class="feature-card">
            <span class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3h5v5M4 20l17-17M21 16v5h-5M15 15l6 6M4 4l5 5"/></svg></span>
            <h3>Preprocessing &amp; augmentations</h3>
            <p>Fully compatible with every <a class="feature-inline-link" href="https://mne.tools/" target="_blank" rel="noopener">MNE</a> preprocessing function and with <a class="feature-inline-link" href="https://eegprep.org/" target="_blank" rel="noopener">EEGPrep</a>, plus exponential standardization and 20+ EEG augmentations.</p>
            <a href="api.html#augmentation" class="feature-link">Augmentation <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></a>
          </div>
          <a href="models/models.html" class="feature-card">
            <span class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M8 6h2a3 3 0 0 1 3 3v0a3 3 0 0 0 3 3M8 18h2a3 3 0 0 0 3-3v0a3 3 0 0 1 3-3"/></svg></span>
            <h3>Curated model zoo</h3>
            <p>EEGNeX, ConvNets, ATCNet, EEGConformer, foundation models. 60+ architectures reproduced from the original papers.</p>
            <span class="feature-link">Browse the zoo <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></span>
          </a>
          <a href="https://mne.tools/" target="_blank" rel="noopener" class="feature-card">
            <span class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 0 7 4.5v.5a2.5 2.5 0 0 0-2 2.4V8a2.5 2.5 0 0 0-2 2.4v.6a2.5 2.5 0 0 0 1 2 2.5 2.5 0 0 0-1 2v.6a2.5 2.5 0 0 0 2 2.4V19a2.5 2.5 0 0 0 2.5 2.5h0A2.5 2.5 0 0 0 10 19V4.5A2.5 2.5 0 0 0 7.5 2h2zM14.5 2A2.5 2.5 0 0 1 17 4.5v.5a2.5 2.5 0 0 1 2 2.4V8a2.5 2.5 0 0 1 2 2.4v.6a2.5 2.5 0 0 1-1 2 2.5 2.5 0 0 1 1 2v.6a2.5 2.5 0 0 1-2 2.4V19a2.5 2.5 0 0 1-2.5 2.5h0A2.5 2.5 0 0 1 14 19V4.5"/></svg></span>
            <h3>Plays well with MNE-Python</h3>
            <p>Native interop with mne.io.Raw and mne.Epochs. Reuse the analysis tools you already know for QC and visualization.</p>
            <span class="feature-link">MNE bridge <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></span>
          </a>
        </div>
      </div>
    </section>

.. ====================================================================

.. CODE / QUICKSTART

.. ====================================================================

.. raw:: html

    <section class="code-section">
      <div class="container">
        <div class="section-head" style="max-width:100%">
          <div class="section-eyebrow">Quickstart</div>
          <h2 class="section-title">Train a model in <em>15 lines.</em></h2>
          <p class="section-sub">From raw motor-imagery EEG to a trained classifier. Each step is a stand-alone module; swap any piece without touching the others.</p>
        </div>
        <div class="code-grid">
          <div class="code-side">
            <ul class="code-steps">
              <li class="code-step active">
                <span class="code-step-num">01</span>
                <span class="code-step-text">
                  <span class="code-step-title">Load the dataset</span>
                  <span class="code-step-desc">Pull BCI Competition IV-2a via MOABB.</span>
                </span>
              </li>
              <li class="code-step">
                <span class="code-step-num">02</span>
                <span class="code-step-text">
                  <span class="code-step-title">Preprocess &amp; window</span>
                  <span class="code-step-desc">Bandpass-filter, then cut motor-imagery trials.</span>
                </span>
              </li>
              <li class="code-step">
                <span class="code-step-num">03</span>
                <span class="code-step-text">
                  <span class="code-step-title">Pick a model</span>
                  <span class="code-step-desc">EEGNeX with 22 channels, 4 classes, 4.5 s @ 250 Hz.</span>
                </span>
              </li>
              <li class="code-step">
                <span class="code-step-num">04</span>
                <span class="code-step-text">
                  <span class="code-step-title">Train</span>
                  <span class="code-step-desc">skorch wrapper with sklearn-style fit().</span>
                </span>
              </li>
            </ul>
          </div>
          <div class="code-window">
            <div class="code-window-bar">
              <div class="code-window-dots"><span></span><span></span><span></span></div>
              <div class="code-window-tabs">
                <span class="code-window-tab active">train_eegnex.py</span>
              </div>
              <div class="code-window-meta">PyTorch · CUDA</div>
            </div>
            <div class="code-window-body">
              <div class="code-gutter"><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div><div>8</div><div>9</div><div>10</div><div>11</div><div>12</div><div>13</div><div>14</div><div>15</div><div>16</div><div>17</div><div>18</div><div>19</div></div>
              <div class="code-content"><div data-step="0"><span class="tok-com"># 1. Load BCI Competition IV-2a (subject 3) via MOABB</span></div><div data-step="0"><span class="tok-kw">from</span> <span class="tok-var">braindecode.datasets</span> <span class="tok-kw">import</span> <span class="tok-cls">MOABBDataset</span></div><div data-step="0"></div><div data-step="0"><span class="tok-var">dataset</span> = <span class="tok-fn">MOABBDataset</span>(<span class="tok-str">"BNCI2014_001"</span>, <span class="tok-attr">subject_ids</span>=[<span class="tok-num">3</span>])</div><div data-step="1"><span class="tok-com"># 2. Bandpass-filter and create event-aligned windows</span></div><div data-step="1"><span class="tok-kw">from</span> <span class="tok-var">braindecode.preprocessing</span> <span class="tok-kw">import</span> (</div><div data-step="1">    <span class="tok-cls">Preprocessor</span>, <span class="tok-fn">preprocess</span>, <span class="tok-fn">create_windows_from_events</span>,</div><div data-step="1">)</div><div data-step="1"></div><div data-step="1"><span class="tok-fn">preprocess</span>(<span class="tok-var">dataset</span>, [<span class="tok-fn">Preprocessor</span>(<span class="tok-str">"filter"</span>, <span class="tok-attr">l_freq</span>=<span class="tok-num">4.</span>, <span class="tok-attr">h_freq</span>=<span class="tok-num">38.</span>)])</div><div data-step="1"><span class="tok-var">windows</span> = <span class="tok-fn">create_windows_from_events</span>(<span class="tok-var">dataset</span>)</div><div data-step="2"><span class="tok-com"># 3. Instantiate a published architecture</span></div><div data-step="2"><span class="tok-kw">from</span> <span class="tok-var">braindecode.models</span> <span class="tok-kw">import</span> <span class="tok-cls">EEGNeX</span></div><div data-step="2"></div><div data-step="2"><span class="tok-var">model</span> = <span class="tok-fn">EEGNeX</span>(<span class="tok-attr">n_chans</span>=<span class="tok-num">22</span>, <span class="tok-attr">n_outputs</span>=<span class="tok-num">4</span>, <span class="tok-attr">n_times</span>=<span class="tok-num">1125</span>)</div><div data-step="3"><span class="tok-com"># 4. Train with the skorch-based EEGClassifier</span></div><div data-step="3"><span class="tok-kw">from</span> <span class="tok-var">braindecode</span> <span class="tok-kw">import</span> <span class="tok-cls">EEGClassifier</span></div><div data-step="3"></div><div data-step="3"><span class="tok-var">clf</span> = <span class="tok-fn">EEGClassifier</span>(<span class="tok-attr">module</span>=<span class="tok-var">model</span>, <span class="tok-attr">max_epochs</span>=<span class="tok-num">10</span>)</div><div data-step="3"><span class="tok-var">clf</span>.<span class="tok-fn">fit</span>(<span class="tok-var">windows</span>, <span class="tok-attr">y</span>=<span class="tok-kw">None</span>)</div></div>
            </div>
            <div class="code-output">
              <span class="ok"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg> Trains on CPU or GPU</span>
              <span>·</span>
              <span>Swap the dataset, model and trainer freely; same API.</span>
              <span style="margin-left:auto"><a href="auto_examples/index.html" style="color:#82aaff">Open the full tutorial →</a></span>
            </div>
          </div>
        </div>
      </div>
    </section>

.. ====================================================================

.. MODEL ZOO BROWSER (interactive)

.. ====================================================================

.. raw:: html

    <section class="section">
      <div class="container">
        <div class="section-head">
          <div class="section-eyebrow">Model zoo</div>
          <h2 class="section-title">A library of <em>published architectures.</em></h2>
          <p class="section-sub">Every model is reproduced from its original paper and ships with a consistent constructor. The shared <code class="bd-mono">EEGModuleMixin</code> takes any of <code class="bd-mono">n_chans</code>, <code class="bd-mono">n_outputs</code>, <code class="bd-mono">n_times</code>, <code class="bd-mono">chs_info</code>, <code class="bd-mono">input_window_seconds</code> or <code class="bd-mono">sfreq</code>; the missing ones are derived for you.</p>
        </div>
        <div class="zoo-toolbar">
          <div class="zoo-search">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
            <input type="text" placeholder="Search models or papers"/>
          </div>
        </div>
        <div class="zoo-grid"><!-- populated by landing.js --></div>
        <div class="zoo-foot">
          <span class="zoo-foot-count">Loading…</span>
          <a href="models/models.html">Browse the full zoo on the Models page <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></a>
        </div>
      </div>
    </section>

.. ====================================================================

.. HUGGING FACE HUB

.. ====================================================================

.. raw:: html

    <section class="section">
      <div class="container">
        <div class="bd-hf">
          <div class="bd-hf-head">
            <div class="bd-hf-marks">
              <img src="_static/braindecode_small.svg" alt="braindecode" style="height:34px;width:auto"/>
              <span class="bd-hf-plus">+</span>
              <img src="_static/hf-logo-with-title.png" alt="Hugging Face" style="height:34px;width:auto"/>
            </div>
            <div>
              <p class="section-eyebrow" style="margin:0 0 .55rem">Hugging Face Hub</p>
              <h2 class="section-title">Pretrained, on the Hub.</h2>
              <p class="section-sub">Every Braindecode model implements <code class="bd-mono">from_pretrained</code> and <code class="bd-mono">push_to_hub</code> &mdash; the same unified API as <code class="bd-mono">transformers</code>. Datasets work the same way.</p>
            </div>
          </div>

          <div class="bd-hf-grid bd-hf-grid-2">
            <div class="bd-hf-card">
              <span class="bd-hf-tag">Models</span>
              <h3>Curated foundation weights</h3>
              <p>BENDR, EEGPT, Signal-JEPA, LaBraM, BIOT and more &mdash; pretrained checkpoints with mapped weights, ready to fine-tune. Curated and benchmarked under <a href="https://huggingface.co/spaces/braindecode/OpenEEGBench" target="_blank" rel="noopener" class="feature-inline-link">OpenEEG-Bench</a>.</p>
              <code class="bd-hf-snippet">model = EEGPT.from_pretrained("braindecode/eegpt")</code>
              <a class="bd-hf-link" href="https://huggingface.co/braindecode" target="_blank" rel="noopener">Visit huggingface.co/braindecode →</a>
            </div>
            <div class="bd-hf-card">
              <span class="bd-hf-tag">Datasets</span>
              <h3>Share &amp; pull EEG datasets</h3>
              <p>Push <code class="bd-mono">WindowsDataset</code>, <code class="bd-mono">EEGWindowsDataset</code> or <code class="bd-mono">RawDataset</code> objects to the Hub with one call. EEGDash mirrors 700+ BIDS-ready EEG/MEG datasets the same way.</p>
              <code class="bd-hf-snippet">ds.push_to_hub("alice/bnci2014_001")</code>
              <a class="bd-hf-link" href="https://eegdash.org/" target="_blank" rel="noopener">Browse EEGDash datasets →</a>
            </div>
          </div>

          <div class="bd-hf-tutorials">
            <div class="bd-hf-tutorials-head">
              <div>
                <span class="bd-hf-tag">Tutorials</span>
                <h3>What you can decode</h3>
              </div>
              <a class="bd-hf-link" href="auto_examples/index.html">Browse all tutorials →</a>
            </div>
            <div class="bd-hf-carousel" tabindex="0">
              <a class="bd-hf-tut" href="auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html">
                <img src="_images/sphx_glr_plot_bcic_iv_2a_moabb_trial_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">Motor imagery</span>
                <span class="bd-hf-tut-title">Trialwise decoding on BCI IV-2a</span>
              </a>
              <a class="bd-hf-tut" href="auto_examples/applied_examples/plot_sleep_staging_chambon2018.html">
                <img src="_images/sphx_glr_plot_sleep_staging_chambon2018_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">Sleep staging</span>
                <span class="bd-hf-tut-title">Chambon 2018 on Sleep Physionet</span>
              </a>
              <a class="bd-hf-tut" href="auto_examples/applied_examples/bcic_iv_4_ecog_trial.html">
                <img src="_images/sphx_glr_bcic_iv_4_ecog_trial_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">ECoG</span>
                <span class="bd-hf-tut-title">Finger-flexion regression on BCI IV-4</span>
              </a>
              <a class="bd-hf-tut" href="auto_examples/advanced_training/plot_finetune_foundation_model.html">
                <img src="_images/sphx_glr_plot_finetune_foundation_model_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">Foundation FT</span>
                <span class="bd-hf-tut-title">Fine-tune Signal-JEPA</span>
              </a>
              <a class="bd-hf-tut" href="auto_examples/model_building/plot_load_pretrained_models.html">
                <img src="_images/sphx_glr_plot_load_pretrained_models_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">Hugging Face</span>
                <span class="bd-hf-tut-title">Load pretrained foundation models</span>
              </a>
              <a class="bd-hf-tut" href="auto_examples/datasets_io/plot_hub_integration.html">
                <img src="_images/sphx_glr_plot_hub_integration_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">Hugging Face</span>
                <span class="bd-hf-tut-title">Push &amp; pull datasets via the Hub</span>
              </a>
              <a class="bd-hf-tut" href="auto_examples/advanced_training/plot_interpretability.html">
                <img src="_images/sphx_glr_plot_interpretability_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">XAI</span>
                <span class="bd-hf-tut-title">Interpretability sanity checks</span>
              </a>
              <a class="bd-hf-tut" href="auto_examples/advanced_training/plot_relative_positioning.html">
                <img src="_images/sphx_glr_plot_relative_positioning_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">Self-supervised</span>
                <span class="bd-hf-tut-title">Relative positioning pretraining</span>
              </a>
              <a class="bd-hf-tut" href="auto_examples/advanced_training/plot_temporal_generalization.html">
                <img src="_images/sphx_glr_plot_temporal_generalization_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">MEG</span>
                <span class="bd-hf-tut-title">Temporal generalization on MEG</span>
              </a>
              <a class="bd-hf-tut" href="auto_examples/advanced_training/plot_data_augmentation.html">
                <img src="_images/sphx_glr_plot_data_augmentation_thumb.png" alt="" loading="lazy" decoding="async"/>
                <span class="bd-hf-tut-tag">Augmentation</span>
                <span class="bd-hf-tut-title">EEG augmentation search</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>

.. ====================================================================

.. PIPELINE

.. ====================================================================

.. raw:: html

    <section class="section pipeline-section">
      <div class="container">
        <div class="section-head">
          <div class="section-eyebrow">Architecture</div>
          <h2 class="section-title">A unified API, <em>nine modules.</em></h2>
          <p class="section-sub">From raw recording to trained checkpoint, the same nine modules cover every task: classification, regression, sleep staging, self-supervised pretraining.</p>
        </div>
        <div class="pipeline">
          <div>
            <div class="pipeline-col-head">
              <span class="pipeline-col-num">PHASE 01</span>
              <span class="pipeline-col-name">Data</span>
            </div>
            <div class="pipeline-col-cards">
              <a class="pipeline-card" href="api.html#datasets">
                <span class="pipeline-card-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/></svg></span>
                <span class="pipeline-card-text">
                  <span class="pipeline-card-name">braindecode.datasets</span>
                  <span class="pipeline-card-desc">MOABB, BIDS, TUH, HBN, sleep corpora.</span>
                </span>
              </a>
              <a class="pipeline-card" href="api.html#preprocessing">
                <span class="pipeline-card-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.5 10 19 14 21 14 12.5 22 3"/></svg></span>
                <span class="pipeline-card-text">
                  <span class="pipeline-card-name">braindecode.preprocessing</span>
                  <span class="pipeline-card-desc">MNE-backed filtering, resampling, windowing.</span>
                </span>
              </a>
              <a class="pipeline-card" href="api.html#samplers">
                <span class="pipeline-card-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3h5v5M4 20l17-17M21 16v5h-5M15 15l6 6M4 4l5 5"/></svg></span>
                <span class="pipeline-card-text">
                  <span class="pipeline-card-name">braindecode.samplers</span>
                  <span class="pipeline-card-desc">Balanced &amp; sequence samplers for windows.</span>
                </span>
              </a>
            </div>
          </div>
          <div>
            <div class="pipeline-col-head">
              <span class="pipeline-col-num">PHASE 02</span>
              <span class="pipeline-col-name">Modeling</span>
            </div>
            <div class="pipeline-col-cards">
              <a class="pipeline-card" href="models/models.html">
                <span class="pipeline-card-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M8 6h2a3 3 0 0 1 3 3v0a3 3 0 0 0 3 3M8 18h2a3 3 0 0 0 3-3v0a3 3 0 0 1 3-3"/></svg></span>
                <span class="pipeline-card-text">
                  <span class="pipeline-card-name">braindecode.models</span>
                  <span class="pipeline-card-desc">60+ architectures, ConvNets to foundation.</span>
                </span>
              </a>
              <a class="pipeline-card" href="api.html#modules">
                <span class="pipeline-card-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 9 5-9 5-9-5 9-5z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></svg></span>
                <span class="pipeline-card-text">
                  <span class="pipeline-card-name">braindecode.modules</span>
                  <span class="pipeline-card-desc">Reusable layers for new architectures.</span>
                </span>
              </a>
              <a class="pipeline-card" href="api.html#augmentation">
                <span class="pipeline-card-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v6m0 8v6M4.93 4.93l4.24 4.24m5.66 5.66 4.24 4.24M2 12h6m8 0h6M4.93 19.07l4.24-4.24m5.66-5.66 4.24-4.24"/></svg></span>
                <span class="pipeline-card-text">
                  <span class="pipeline-card-name">braindecode.augmentation</span>
                  <span class="pipeline-card-desc">20+ EEG-specific data augmentations.</span>
                </span>
              </a>
            </div>
          </div>
          <div>
            <div class="pipeline-col-head">
              <span class="pipeline-col-num">PHASE 03</span>
              <span class="pipeline-col-name">Train &amp; evaluate</span>
            </div>
            <div class="pipeline-col-cards">
              <a class="pipeline-card" href="api.html#training">
                <span class="pipeline-card-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a4 4 0 0 0 5 5l-9 9a2 2 0 1 1-2.8-2.8z"/></svg></span>
                <span class="pipeline-card-text">
                  <span class="pipeline-card-name">braindecode.training</span>
                  <span class="pipeline-card-desc">Cropped decoding, scoring callbacks.</span>
                </span>
              </a>
              <a class="pipeline-card" href="api.html#classifier">
                <span class="pipeline-card-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 4 20 12 6 20 6 4"/></svg></span>
                <span class="pipeline-card-text">
                  <span class="pipeline-card-name">EEGClassifier / EEGRegressor</span>
                  <span class="pipeline-card-desc">skorch wrappers with .fit() / .predict().</span>
                </span>
              </a>
              <a class="pipeline-card" href="api.html#visualization">
                <span class="pipeline-card-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3-9 4 18 3-9h4"/></svg></span>
                <span class="pipeline-card-text">
                  <span class="pipeline-card-name">braindecode.visualization</span>
                  <span class="pipeline-card-desc">Topomaps, gradient plots, interpretability.</span>
                </span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>

.. ====================================================================

.. ECOSYSTEM / IN GOOD COMPANY

.. ====================================================================

.. raw:: html

    <section class="section" id="ecosystem">
      <div class="container">
        <div class="section-head">
          <div class="section-eyebrow">In the ecosystem</div>
          <h2 class="section-title">Built and shared <em>with neighbors.</em></h2>
          <p class="section-sub">Open neural decoding doesn't live alone. Braindecode plugs into a wider open-source neuroscience stack, ships sibling projects that round out the data story, and is built on top of by other research toolkits.</p>
        </div>
        <div class="ecosystem-grid">
          <a class="ecosystem-card" href="https://eegdash.org/" target="_blank" rel="noopener">
            <div class="ecosystem-tag">Sister project</div>
            <div class="ecosystem-name">EEGDash</div>
            <div class="ecosystem-desc">Data-sharing archive with <strong>700+ BIDS-ready</strong> EEG/MEG/iEEG/fNIRS/EMG recordings from collaborating labs.</div>
          </a>
          <a class="ecosystem-card" href="https://moabb.neurotechx.com/" target="_blank" rel="noopener">
            <div class="ecosystem-tag">Benchmark partner</div>
            <div class="ecosystem-name">MOABB</div>
            <div class="ecosystem-desc">Reproducible motor-imagery benchmarking. Braindecode wraps every MOABB dataset out of the box.</div>
          </a>
          <a class="ecosystem-card" href="https://mne.tools/" target="_blank" rel="noopener">
            <div class="ecosystem-tag">Foundation</div>
            <div class="ecosystem-name">MNE-Python</div>
            <div class="ecosystem-desc">Braindecode preprocessing builds on MNE; any <code>mne.io.Raw</code> or <code>mne.Epochs</code> works as input.</div>
          </a>
          <a class="ecosystem-card" href="https://facebookresearch.github.io/neuroai/" target="_blank" rel="noopener">
            <div class="ecosystem-card-head">
              <span class="ecosystem-tag">FAIR ecosystem</span>
              <span class="ecosystem-meta-mark">Meta</span>
            </div>
            <div class="ecosystem-name">Meta Neuro AI</div>
            <div class="ecosystem-desc">FAIR's Python suite for Neuro-AI: discovery, preprocessing and deep learning across neural and stimulus data. Ships <strong>NeuralTrain</strong>, which depends on Braindecode.</div>
          </a>
          <a class="ecosystem-card" href="https://pypi.org/project/neuraltrain/" target="_blank" rel="noopener">
            <div class="ecosystem-card-head">
              <span class="ecosystem-tag">Used by</span>
              <span class="ecosystem-meta-mark">Meta</span>
            </div>
            <div class="ecosystem-name">NeuralTrain</div>
            <div class="ecosystem-desc">Meta FAIR's lightweight deep-learning library for M/EEG. Lists <code>braindecode</code> under its <code>models</code> extras and pulls in Braindecode's <code>LABRAM</code> and <code>EEGPT</code> backbones.</div>
          </a>
          <a class="ecosystem-card" href="https://github.com/aeon-toolkit/aeon-neuro" target="_blank" rel="noopener">
            <div class="ecosystem-tag">Used by</div>
            <div class="ecosystem-name">aeon-neuro</div>
            <div class="ecosystem-desc">aeon-toolkit's neuroscience extension wraps Braindecode models for time-series classification on EEG.</div>
          </a>
        </div>
        <p class="ecosystem-foot">
          Building on Braindecode? <a href="https://github.com/braindecode/braindecode/network/dependents" target="_blank" rel="noopener">See the full list of dependent projects on GitHub →</a>
        </p>
      </div>
    </section>

.. ====================================================================

.. CTA + FOOTER

.. ====================================================================

.. raw:: html

    <section class="section">
      <div class="container">
        <div class="cta-banner">
          <h2>Ready to <em>brain decode</em> something?</h2>
          <p>Install in one line, follow a tutorial in fifteen, publish an experiment by Friday.</p>
          <div class="cta-row">
            <a href="install/install.html" class="btn btn-primary">Get started <span class="btn-arrow">→</span></a>
            <a href="auto_examples/index.html" class="btn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
              Read the docs
            </a>
            <a href="https://github.com/braindecode/braindecode" target="_blank" rel="noopener" class="btn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-4.3 1.4-4.3-2.5-6-3m12 5v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.2 4.2 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"/></svg>
              Star on GitHub
            </a>
          </div>
        </div>
      </div>
    </section>

    <footer class="site-footer">
      <div class="container">
        <div class="site-footer-cols">
          <div>
            <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.7rem;color:var(--bd-ink)">
              <img src="_static/braindecode_small.svg" alt="" width="24" height="24" style="display:block;width:24px;height:24px;object-fit:contain"/>
              <b style="font-size:1rem;letter-spacing:-.02em">braindecode</b>
            </div>
            <p class="site-footer-blurb">Open-source toolbox for decoding raw EEG, ECoG and MEG with deep learning. Built on PyTorch, MNE-Python and skorch.</p>
          </div>
          <div>
            <h5>Get started</h5>
            <ul>
              <li><a href="install/install.html">Install</a></li>
              <li><a href="auto_examples/index.html">Tutorials</a></li>
              <li><a href="models/models.html">Models</a></li>
              <li><a href="api.html">API reference</a></li>
            </ul>
          </div>
          <div>
            <h5>Community</h5>
            <ul>
              <li><a href="https://github.com/braindecode/braindecode" target="_blank" rel="noopener">GitHub</a></li>
              <li><a href="https://github.com/braindecode/braindecode/discussions" target="_blank" rel="noopener">Discussions</a></li>
              <li><a href="https://github.com/braindecode/braindecode/graphs/contributors" target="_blank" rel="noopener">Contributors</a></li>
              <li><a href="cite.html">Cite</a></li>
            </ul>
          </div>
          <div>
            <h5>Resources</h5>
            <ul>
              <li><a href="whats_new.html">What's new</a></li>
              <li><a href="https://github.com/braindecode/braindecode/issues" target="_blank" rel="noopener">Issues</a></li>
              <li><a href="help.html">Help &amp; support</a></li>
            </ul>
          </div>
          <div>
            <h5>Friends</h5>
            <ul>
              <li><a href="https://eegdash.org/" target="_blank" rel="noopener">EEGDash</a></li>
              <li><a href="https://mne.tools/" target="_blank" rel="noopener">MNE-Python</a></li>
              <li><a href="https://moabb.neurotechx.com/" target="_blank" rel="noopener">MOABB</a></li>
              <li><a href="https://pytorch.org/" target="_blank" rel="noopener">PyTorch</a></li>
              <li><a href="https://skorch.readthedocs.io/" target="_blank" rel="noopener">skorch</a></li>
            </ul>
          </div>
        </div>
        <div class="site-footer-bottom">
          <span>© 2017–2026 The Braindecode developers · BSD-3-Clause License</span>
          <span>Made with care across UC San Diego, Inria, Uni Freiburg, and Donders.</span>
        </div>
      </div>
    </footer>

    </div>
    <!-- /.bd-landing -->

.. ----------------------------------------------------------------------

.. Sidebar toctrees (hidden, captioned)

.. ----------------------------------------------------------------------

.. toctree::
    :hidden:
    :caption: Get Started

    Install <install/install>
    Tutorial and Examples <auto_examples/index>
    Models <models/models>

.. toctree::
    :hidden:
    :caption: Documentation

    API <api>
    What's new <whats_new>

.. toctree::
    :hidden:
    :caption: Community

    Cite <cite>
    Get help <help>
