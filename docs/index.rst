:html_theme.sidebar_secondary.remove: true

.. meta::
    :description: Braindecode is the open-source PyTorch library for decoding raw EEG, MEG, iEEG, ECoG, fNIRS, and sEMG signals. Load continuous recordings, map any sensor montage, and train or fine-tune 100+ published architectures including 48 foundation models.
    :keywords: braindecode, EEG decoding, MEG, iEEG, ECoG, sEEG, fNIRS, sEMG, neural foundation models, deep learning, PyTorch, MNE-Python, MOABB, EEGDash

.. raw:: html

    <div class="bd-landing">

###############################################################################
 Braindecode — Open-source PyTorch library for EEG, iEEG, MEG, fNIRS, and sEMG
###############################################################################

.. raw:: html

    <section class="hero">
      <div class="container">
        <div class="hero-grid">
          <div class="hero-copy">
            <p class="hero-title">From raw brain signals to predictions.</p>
            <p class="hero-lede">
              The open-source PyTorch library for decoding EEG, iEEG, MEG, fNIRS, and sEMG. Load continuous recordings, map any sensor montage, and train or fine-tune 103 peer-reviewed architectures.
            </p>
            <div class="hero-cta-row">
              <a href="install/install.html" class="btn btn-primary">Get started <span class="btn-arrow" aria-hidden="true">→</span></a>
              <a href="#model-zoo" class="btn">Explore 103 models</a>
              <span class="hero-install-wrap">
                <button type="button" class="hero-install bd-groove" data-copy="pip install braindecode" aria-label="Copy pip install braindecode">
                  <span class="prompt" aria-hidden="true">$</span>
                  <code>pip install braindecode</code>
                  <span class="copy-icon" aria-hidden="true">
                    <svg class="icon-clipboard" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    <svg class="icon-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
                  </span>
                </button>
                <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
              </span>
            </div>
            <div class="proof-bar bd-mono">
              <a href="models/models_table.html"><b>103</b> ARCHITECTURES</a>
              <span class="proof-div" aria-hidden="true"></span>
              <a href="#foundation-hub"><b>48</b> FOUNDATION MODELS</a>
              <span class="proof-div" aria-hidden="true"></span>
              <a href="https://pepy.tech/projects/braindecode" target="_blank" rel="noopener"><b id="bd-downloads-num">1.4M</b> DOWNLOADS</a>
              <span class="proof-div" aria-hidden="true"></span>
              <a href="cite.html"><b>HBM 2017</b> · BSD-3-CLAUSE</a>
            </div>
          </div>

          <div class="hero-stage bd-pane">
            <div class="stage-head">
              <fieldset class="stage-tabs" aria-label="Select neurophysiological decoding view">
                <label class="stage-tab">
                  <input type="radio" name="bd-hero-tab" value="mi" class="sr-only" checked/>
                  <span>Motor Imagery EEG</span>
                </label>
                <label class="stage-tab">
                  <input type="radio" name="bd-hero-tab" value="sleep" class="sr-only"/>
                  <span>Sleep PSG</span>
                </label>
                <label class="stage-tab">
                  <input type="radio" name="bd-hero-tab" value="szcore" class="sr-only"/>
                  <span>Clinical SzCORE</span>
                </label>
              </fieldset>
              <span class="stage-spec stage-spec-mi bd-mono">X: (22, 1000) @ 250 Hz</span>
              <span class="stage-spec stage-spec-sleep bd-mono">X: (4, 3840) @ 128 Hz</span>
              <span class="stage-spec stage-spec-szcore bd-mono">X: (18, 15360) @ 256 Hz</span>
            </div>

            <div class="stage-panel stage-panel-mi">
              <svg class="stage-svg" viewBox="0 0 540 248" role="img" aria-label="Contralateral C3 and CP3 mu desynchronization decoded by EEGNeX as Right Hand motor imagery at 0.91 probability">
                <rect x="0.5" y="0.5" width="539" height="247" rx="10" class="stage-canvas"/>
                <g class="stage-ch-labels bd-mono">
                  <text x="14" y="40">C3</text>
                  <text x="14" y="84">Cz</text>
                  <text x="14" y="128">C4</text>
                  <text x="14" y="172">CP3</text>
                  <text x="14" y="216">CP4</text>
                </g>
                <g class="stage-rays" fill="none">
                  <path d="M238 36 C276 36, 286 124, 304 124"/>
                  <path d="M238 80 C274 80, 286 124, 304 124"/>
                  <path d="M238 124 L304 124"/>
                  <path d="M238 168 C274 168, 286 124, 304 124"/>
                  <path d="M238 212 C276 212, 286 124, 304 124"/>
                </g>
                <rect x="126" y="14" width="112" height="214" rx="6" class="stage-epoch"/>
                <text x="182" y="241" text-anchor="middle" class="stage-epoch-lbl bd-mono">EPOCH 4.0s · ERD</text>
                <g fill="none" stroke-linecap="round" stroke-linejoin="round">
                  <path class="trace-hi" d="M54.0 32.0 L56.0 30.2 L58.0 30.0 L60.0 30.8 L62.0 31.5 L64.0 31.5 L66.0 30.7 L68.0 29.6 L70.0 29.2 L72.0 30.4 L74.0 33.4 L76.0 37.8 L78.0 42.3 L80.0 45.5 L82.0 46.3 L84.0 44.6 L86.0 40.7 L88.0 36.0 L90.0 31.8 L92.0 29.2 L94.0 28.5 L96.0 29.1 L98.0 30.1 L100.0 30.6 L102.0 30.3 L104.0 29.4 L106.0 28.7 L108.0 29.3 L110.0 31.6 L112.0 35.6 L114.0 40.4 L116.0 44.5 L118.0 46.7 L120.0 46.3 L122.0 43.4 L124.0 39.1 L126.0 35.2 L128.0 34.1 L130.0 33.7 L132.0 33.8 L134.0 34.3 L136.0 34.7 L138.0 34.8 L140.0 34.7 L142.0 34.5 L144.0 34.5 L146.0 35.1 L148.0 36.4 L150.0 38.1 L152.0 39.9 L154.0 41.1 L156.0 41.4 L158.0 40.8 L160.0 39.5 L162.0 37.9 L164.0 36.5 L166.0 35.6 L168.0 35.4 L170.0 35.7 L172.0 36.1 L174.0 36.2 L176.0 36.1 L178.0 35.7 L180.0 35.4 L182.0 35.6 L184.0 36.5 L186.0 37.9 L188.0 39.5 L190.0 40.8 L192.0 41.4 L194.0 41.1 L196.0 39.9 L198.0 38.1 L200.0 36.4 L202.0 35.1 L204.0 34.5 L206.0 34.5 L208.0 34.7 L210.0 34.8 L212.0 34.7 L214.0 34.3 L216.0 33.8 L218.0 33.7 L220.0 34.1 L222.0 35.2 L224.0 36.7 L226.0 38.2 L228.0 39.2 L230.0 39.3 L232.0 38.4 L234.0 36.9 L236.0 35.1 L238.0 33.6 L240.0 29.3 L242.0 28.7 L244.0 29.4 L246.0 30.3 L248.0 30.6 L250.0 30.1 L252.0 29.1 L254.0 28.5 L256.0 29.2 L258.0 31.8 L260.0 36.0 L262.0 40.7 L264.0 44.6 L266.0 46.3 L268.0 45.5 L270.0 42.3 L272.0 37.8 L274.0 33.4 L276.0 30.4 L278.0 29.2 L280.0 29.6 L282.0 30.7 L284.0 31.5 L286.0 31.5 L288.0 30.8 L290.0 30.0 L292.0 30.2 L294.0 32.0"/>
                  <path class="trace-muted" d="M54.0 68.9 L56.0 66.9 L58.0 67.5 L60.0 70.5 L62.0 74.9 L64.0 79.3 L66.0 82.4 L68.0 83.7 L70.0 83.6 L72.0 82.8 L74.0 82.2 L76.0 82.5 L78.0 83.6 L80.0 84.7 L82.0 84.9 L84.0 83.3 L86.0 79.9 L88.0 75.3 L90.0 70.9 L92.0 67.9 L94.0 67.5 L96.0 69.6 L98.0 73.7 L100.0 78.4 L102.0 82.3 L104.0 84.6 L106.0 85.2 L108.0 84.7 L110.0 84.1 L112.0 84.0 L114.0 84.9 L116.0 86.2 L118.0 87.0 L120.0 86.3 L122.0 83.7 L124.0 79.6 L126.0 74.9 L128.0 71.1 L130.0 69.4 L132.0 70.4 L134.0 73.7 L136.0 78.2 L138.0 82.6 L140.0 85.6 L142.0 86.9 L144.0 86.7 L146.0 85.9 L148.0 85.4 L150.0 85.7 L152.0 86.8 L154.0 87.8 L156.0 87.7 L158.0 85.9 L160.0 82.2 L162.0 77.5 L164.0 72.9 L166.0 70.0 L168.0 69.6 L170.0 71.7 L172.0 75.7 L174.0 80.1 L176.0 83.7 L178.0 85.7 L180.0 85.9 L182.0 85.1 L184.0 84.2 L186.0 84.0 L188.0 84.7 L190.0 85.7 L192.0 86.2 L194.0 85.1 L196.0 82.2 L198.0 77.7 L200.0 72.9 L202.0 69.0 L204.0 67.3 L206.0 68.3 L208.0 71.6 L210.0 76.0 L212.0 80.1 L214.0 82.9 L216.0 84.0 L218.0 83.6 L220.0 82.7 L222.0 82.2 L224.0 82.6 L226.0 83.7 L228.0 84.6 L230.0 84.5 L232.0 82.6 L234.0 78.9 L236.0 74.2 L238.0 69.8 L240.0 67.2 L242.0 67.1 L244.0 69.6 L246.0 73.8 L248.0 78.4 L250.0 82.1 L252.0 84.1 L254.0 84.5 L256.0 83.9 L258.0 83.2 L260.0 83.3 L262.0 84.3 L264.0 85.6 L266.0 86.2 L268.0 85.3 L270.0 82.5 L272.0 78.2 L274.0 73.6 L276.0 70.1 L278.0 68.8 L280.0 70.2 L282.0 73.8 L284.0 78.4 L286.0 82.7 L288.0 85.5 L290.0 86.6 L292.0 86.3 L294.0 85.5"/>
                  <path class="trace-muted" d="M54.0 116.4 L56.0 115.8 L58.0 116.8 L60.0 119.7 L62.0 124.3 L64.0 129.2 L66.0 133.1 L68.0 134.8 L70.0 133.9 L72.0 130.7 L74.0 126.4 L76.0 122.5 L78.0 119.8 L80.0 118.9 L82.0 119.3 L84.0 120.1 L86.0 120.5 L88.0 120.0 L90.0 118.8 L92.0 117.9 L94.0 118.1 L96.0 120.2 L98.0 124.1 L100.0 129.1 L102.0 133.7 L104.0 136.5 L106.0 136.7 L108.0 134.5 L110.0 130.5 L112.0 126.1 L114.0 122.7 L116.0 120.9 L118.0 120.7 L120.0 121.4 L122.0 121.9 L124.0 121.7 L126.0 120.6 L128.0 119.3 L130.0 118.7 L132.0 119.8 L134.0 122.8 L136.0 127.2 L138.0 132.0 L140.0 135.6 L142.0 136.9 L144.0 135.6 L146.0 132.1 L148.0 127.6 L150.0 123.4 L152.0 120.7 L154.0 119.7 L156.0 119.9 L158.0 120.5 L160.0 120.5 L162.0 119.7 L164.0 118.2 L166.0 117.1 L168.0 117.2 L170.0 119.2 L172.0 123.1 L174.0 127.8 L176.0 132.1 L178.0 134.5 L180.0 134.4 L182.0 131.8 L184.0 127.6 L186.0 123.1 L188.0 119.7 L190.0 117.9 L192.0 117.7 L194.0 118.3 L196.0 118.8 L198.0 118.5 L200.0 117.3 L202.0 116.0 L204.0 115.6 L206.0 116.8 L208.0 120.0 L210.0 124.7 L212.0 129.5 L214.0 133.1 L216.0 134.3 L218.0 133.0 L220.0 129.5 L222.0 125.1 L224.0 121.3 L226.0 118.9 L228.0 118.2 L230.0 118.7 L232.0 119.5 L234.0 119.7 L236.0 119.1 L238.0 117.9 L240.0 117.0 L242.0 117.5 L244.0 120.0 L246.0 124.2 L248.0 129.2 L250.0 133.6 L252.0 136.1 L254.0 135.9 L256.0 133.4 L258.0 129.3 L260.0 125.0 L262.0 121.9 L264.0 120.4 L266.0 120.5 L268.0 121.2 L270.0 121.7 L272.0 121.4 L274.0 120.3 L276.0 119.1 L278.0 118.8 L280.0 120.2 L282.0 123.5 L284.0 128.2 L286.0 132.9 L288.0 136.3 L290.0 137.3 L292.0 135.6 L294.0 131.9"/>
                  <path class="trace-hi" d="M54.0 172.0 L56.0 172.2 L58.0 171.8 L60.0 171.6 L62.0 172.3 L64.0 173.7 L66.0 175.3 L68.0 175.9 L70.0 174.7 L72.0 171.6 L74.0 167.0 L76.0 162.2 L78.0 158.7 L80.0 157.6 L82.0 159.0 L84.0 162.6 L86.0 166.9 L88.0 170.8 L90.0 173.2 L92.0 174.0 L94.0 173.6 L96.0 173.1 L98.0 173.2 L100.0 174.2 L102.0 175.7 L104.0 176.8 L106.0 176.3 L108.0 173.9 L110.0 169.7 L112.0 164.7 L114.0 160.3 L116.0 157.8 L118.0 157.9 L120.0 160.4 L122.0 164.4 L124.0 168.5 L126.0 169.7 L128.0 170.1 L130.0 170.0 L132.0 169.7 L134.0 169.5 L136.0 169.6 L138.0 169.9 L140.0 170.3 L142.0 170.3 L144.0 169.7 L146.0 168.3 L148.0 166.5 L150.0 164.7 L152.0 163.3 L154.0 162.8 L156.0 163.3 L158.0 164.5 L160.0 166.0 L162.0 167.2 L164.0 168.0 L166.0 168.1 L168.0 168.0 L170.0 167.7 L172.0 167.8 L174.0 168.1 L176.0 168.7 L178.0 169.0 L180.0 168.8 L182.0 167.9 L184.0 166.4 L186.0 164.7 L188.0 163.2 L190.0 162.4 L192.0 162.6 L194.0 163.7 L196.0 165.3 L198.0 166.9 L200.0 168.0 L202.0 168.6 L204.0 168.7 L206.0 168.6 L208.0 168.7 L210.0 169.0 L212.0 169.7 L214.0 170.3 L216.0 170.5 L218.0 170.0 L220.0 168.9 L222.0 167.3 L224.0 165.7 L226.0 164.6 L228.0 164.4 L230.0 165.2 L232.0 166.6 L234.0 168.3 L236.0 169.7 L238.0 170.5 L240.0 173.6 L242.0 173.3 L244.0 172.9 L246.0 173.2 L248.0 174.4 L250.0 175.9 L252.0 176.8 L254.0 176.2 L256.0 173.5 L258.0 169.1 L260.0 164.2 L262.0 160.0 L264.0 158.0 L266.0 158.5 L268.0 161.4 L270.0 165.6 L272.0 169.6 L274.0 172.4 L276.0 173.5 L278.0 173.3 L280.0 172.7 L282.0 172.4 L284.0 173.1 L286.0 174.4 L288.0 175.6 L290.0 175.6 L292.0 173.8 L294.0 169.9"/>
                  <path class="trace-muted" d="M54.0 224.6 L56.0 224.3 L58.0 221.7 L60.0 217.7 L62.0 213.8 L64.0 211.0 L66.0 209.8 L68.0 209.8 L70.0 210.3 L72.0 210.3 L74.0 209.4 L76.0 207.8 L78.0 206.2 L80.0 205.9 L82.0 207.5 L84.0 211.1 L86.0 215.9 L88.0 220.7 L90.0 223.9 L92.0 224.7 L94.0 222.9 L96.0 219.2 L98.0 214.9 L100.0 211.3 L102.0 209.2 L104.0 208.6 L106.0 208.9 L108.0 209.1 L110.0 208.5 L112.0 207.0 L114.0 205.2 L116.0 204.1 L118.0 204.6 L120.0 207.3 L122.0 211.6 L124.0 216.5 L126.0 220.6 L128.0 222.5 L130.0 221.9 L132.0 218.9 L134.0 214.8 L136.0 210.8 L138.0 208.0 L140.0 206.8 L142.0 206.8 L144.0 207.2 L146.0 207.1 L148.0 206.1 L150.0 204.4 L152.0 203.0 L154.0 202.8 L156.0 204.6 L158.0 208.4 L160.0 213.4 L162.0 218.2 L164.0 221.4 L166.0 222.1 L168.0 220.3 L170.0 216.7 L172.0 212.6 L174.0 209.4 L176.0 207.6 L178.0 207.3 L180.0 207.8 L182.0 208.2 L184.0 207.7 L186.0 206.5 L188.0 204.9 L190.0 204.2 L192.0 205.1 L194.0 208.1 L196.0 212.8 L198.0 218.0 L200.0 222.1 L202.0 224.1 L204.0 223.5 L206.0 220.6 L208.0 216.6 L210.0 212.8 L212.0 210.3 L214.0 209.4 L216.0 209.6 L218.0 210.1 L220.0 210.0 L222.0 209.0 L224.0 207.4 L226.0 206.1 L228.0 206.1 L230.0 208.0 L232.0 211.9 L234.0 216.9 L236.0 221.6 L238.0 224.5 L240.0 225.0 L242.0 222.8 L244.0 219.0 L246.0 214.8 L248.0 211.5 L250.0 209.7 L252.0 209.3 L254.0 209.6 L256.0 209.7 L258.0 209.0 L260.0 207.4 L262.0 205.7 L264.0 204.8 L266.0 205.7 L268.0 208.6 L270.0 213.1 L272.0 218.0 L274.0 221.8 L276.0 223.3 L278.0 222.2 L280.0 219.0 L282.0 214.7 L284.0 210.8 L286.0 208.2 L288.0 207.2 L290.0 207.3 L292.0 207.6 L294.0 207.4"/>
                </g>
                <rect x="292" y="106" width="68" height="36" rx="8" class="stage-model-box"/>
                <text x="326" y="128" text-anchor="middle" class="stage-model-txt bd-mono">EEGNeX</text>
                <text x="374" y="32" class="stage-out-head bd-mono">POSTERIOR</text>
                <g class="stage-bars">
                  <text x="374" y="60" class="bar-lbl-hi">Right Hand</text>
                  <text x="524" y="60" text-anchor="end" class="bar-val-hi bd-mono">0.91</text>
                  <rect x="374" y="68" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="68" width="136.5" height="8" rx="4" class="bar-fill-hi"/>

                  <text x="374" y="108" class="bar-lbl">Left Hand</text>
                  <text x="524" y="108" text-anchor="end" class="bar-val bd-mono">0.04</text>
                  <rect x="374" y="116" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="116" width="10" height="8" rx="4" class="bar-fill"/>

                  <text x="374" y="156" class="bar-lbl">Feet</text>
                  <text x="524" y="156" text-anchor="end" class="bar-val bd-mono">0.03</text>
                  <rect x="374" y="164" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="164" width="8" height="8" rx="4" class="bar-fill"/>

                  <text x="374" y="204" class="bar-lbl">Tongue</text>
                  <text x="524" y="204" text-anchor="end" class="bar-val bd-mono">0.02</text>
                  <rect x="374" y="212" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="212" width="7" height="8" rx="4" class="bar-fill"/>
                </g>
              </svg>
            </div>

            <div class="stage-panel stage-panel-sleep">
              <svg class="stage-svg" viewBox="0 0 540 248" role="img" aria-label="Polysomnography K-complex and 13 Hz sleep spindle decoded by SleepFM as N2 stage at 0.88 probability">
                <rect x="0.5" y="0.5" width="539" height="247" rx="10" class="stage-canvas"/>
                <g class="stage-ch-labels bd-mono">
                  <text x="10" y="46">Fpz-Cz</text>
                  <text x="10" y="98">Pz-Oz</text>
                  <text x="10" y="150">EOG-L</text>
                  <text x="10" y="202">EMG</text>
                </g>
                <g class="stage-rays" fill="none">
                  <path d="M238 42 C276 42, 286 124, 304 124"/>
                  <path d="M238 94 C274 94, 286 124, 304 124"/>
                  <path d="M238 146 C274 146, 286 124, 304 124"/>
                  <path d="M238 198 C276 198, 286 124, 304 124"/>
                </g>
                <rect x="126" y="16" width="112" height="208" rx="6" class="stage-epoch"/>
                <text x="182" y="239" text-anchor="middle" class="stage-epoch-lbl bd-mono">30s PSG · SPINDLE</text>
                <g fill="none" stroke-linecap="round" stroke-linejoin="round">
                  <path class="trace-hi" d="M54.0 42.0 L56.0 41.6 L58.0 41.3 L60.0 40.9 L62.0 40.6 L64.0 40.3 L66.0 40.0 L68.0 39.8 L70.0 39.6 L72.0 39.4 L74.0 39.3 L76.0 39.2 L78.0 39.2 L80.0 39.2 L82.0 39.3 L84.0 39.4 L86.0 39.6 L88.0 39.8 L90.0 40.0 L92.0 40.3 L94.0 40.6 L96.0 40.9 L98.0 41.3 L100.0 41.6 L102.0 42.0 L104.0 42.4 L106.0 42.7 L108.0 43.1 L110.0 43.4 L112.0 43.7 L114.0 44.0 L116.0 44.2 L118.0 44.4 L120.0 44.6 L122.0 44.7 L124.0 44.8 L126.0 44.8 L128.0 44.9 L130.0 45.0 L132.0 45.4 L134.0 46.3 L136.0 47.9 L138.0 50.4 L140.0 53.4 L142.0 56.0 L144.0 57.1 L146.0 56.0 L148.0 52.4 L150.0 47.2 L152.0 41.3 L154.0 36.0 L156.0 31.9 L158.0 29.4 L160.0 28.5 L162.0 28.5 L164.0 29.0 L166.0 30.1 L168.0 32.3 L170.0 35.3 L172.0 37.4 L174.0 37.1 L176.0 35.6 L178.0 36.2 L180.0 40.3 L182.0 44.0 L184.0 42.4 L186.0 36.5 L188.0 33.7 L190.0 39.1 L192.0 47.2 L194.0 48.4 L196.0 40.8 L198.0 33.9 L200.0 36.7 L202.0 46.1 L204.0 50.9 L206.0 46.4 L208.0 39.2 L210.0 38.3 L212.0 43.7 L214.0 48.4 L216.0 47.7 L218.0 44.0 L220.0 42.2 L222.0 43.5 L224.0 45.6 L226.0 46.0 L228.0 44.9 L230.0 43.9 L232.0 43.7 L234.0 44.0 L236.0 43.9 L238.0 43.5 L240.0 43.0 L242.0 42.6 L244.0 42.3 L246.0 42.0 L248.0 41.7 L250.0 41.3 L252.0 40.9 L254.0 40.6 L256.0 40.3 L258.0 40.0 L260.0 39.8 L262.0 39.6 L264.0 39.4 L266.0 39.3 L268.0 39.2 L270.0 39.2 L272.0 39.2 L274.0 39.3 L276.0 39.4 L278.0 39.6 L280.0 39.8 L282.0 40.0 L284.0 40.3 L286.0 40.6 L288.0 40.9 L290.0 41.3 L292.0 41.6 L294.0 42.0"/>
                  <path class="trace-hi" d="M54.0 91.6 L56.0 91.5 L58.0 91.3 L60.0 91.2 L62.0 91.2 L64.0 91.2 L66.0 91.3 L68.0 91.4 L70.0 91.5 L72.0 91.7 L74.0 91.9 L76.0 92.2 L78.0 92.5 L80.0 92.8 L82.0 93.1 L84.0 93.5 L86.0 93.9 L88.0 94.2 L90.0 94.6 L92.0 94.9 L94.0 95.3 L96.0 95.6 L98.0 95.9 L100.0 96.1 L102.0 96.4 L104.0 96.5 L106.0 96.7 L108.0 96.8 L110.0 96.8 L112.0 96.8 L114.0 96.7 L116.0 96.6 L118.0 96.5 L120.0 96.3 L122.0 96.1 L124.0 95.8 L126.0 95.5 L128.0 95.2 L130.0 95.0 L132.0 94.8 L134.0 94.8 L136.0 95.1 L138.0 95.8 L140.0 96.9 L142.0 98.3 L144.0 99.4 L146.0 100.0 L148.0 99.6 L150.0 97.9 L152.0 95.3 L154.0 92.3 L156.0 89.3 L158.0 87.1 L160.0 85.7 L162.0 85.0 L164.0 84.5 L166.0 84.8 L168.0 86.3 L170.0 88.9 L172.0 90.8 L174.0 90.5 L176.0 89.1 L178.0 90.0 L180.0 94.3 L182.0 98.2 L184.0 96.8 L186.0 91.0 L188.0 88.4 L190.0 93.8 L192.0 101.9 L194.0 103.0 L196.0 95.3 L198.0 88.3 L200.0 90.9 L202.0 100.0 L204.0 104.6 L206.0 99.8 L208.0 92.3 L210.0 91.0 L212.0 96.1 L214.0 100.5 L216.0 99.5 L218.0 95.4 L220.0 93.2 L222.0 94.3 L224.0 96.0 L226.0 96.1 L228.0 94.8 L230.0 93.6 L232.0 93.3 L234.0 93.4 L236.0 93.3 L238.0 92.9 L240.0 92.4 L242.0 92.0 L244.0 91.8 L246.0 91.7 L248.0 91.5 L250.0 91.3 L252.0 91.2 L254.0 91.2 L256.0 91.2 L258.0 91.3 L260.0 91.4 L262.0 91.5 L264.0 91.7 L266.0 91.9 L268.0 92.2 L270.0 92.5 L272.0 92.8 L274.0 93.1 L276.0 93.5 L278.0 93.9 L280.0 94.2 L282.0 94.6 L284.0 94.9 L286.0 95.3 L288.0 95.6 L290.0 95.9 L292.0 96.1 L294.0 96.4"/>
                  <path class="trace-muted" d="M54.0 143.2 L56.0 142.9 L58.0 142.8 L60.0 142.6 L62.0 142.5 L64.0 142.4 L66.0 142.4 L68.0 142.4 L70.0 142.4 L72.0 142.4 L74.0 142.5 L76.0 142.5 L78.0 142.6 L80.0 142.7 L82.0 142.8 L84.0 142.8 L86.0 142.9 L88.0 142.9 L90.0 143.0 L92.0 143.0 L94.0 143.0 L96.0 143.0 L98.0 143.0 L100.0 143.0 L102.0 142.9 L104.0 142.9 L106.0 142.8 L108.0 142.8 L110.0 142.7 L112.0 142.7 L114.0 142.6 L116.0 142.6 L118.0 142.6 L120.0 142.7 L122.0 142.7 L124.0 142.8 L126.0 143.0 L128.0 143.1 L130.0 143.4 L132.0 143.6 L134.0 143.9 L136.0 144.2 L138.0 144.5 L140.0 144.9 L142.0 145.3 L144.0 145.7 L146.0 146.2 L148.0 146.6 L150.0 147.1 L152.0 147.5 L154.0 147.9 L156.0 148.4 L158.0 148.8 L160.0 149.1 L162.0 149.5 L164.0 149.8 L166.0 150.1 L168.0 150.3 L170.0 150.5 L172.0 150.6 L174.0 150.7 L176.0 150.8 L178.0 150.8 L180.0 150.8 L182.0 150.7 L184.0 150.6 L186.0 150.5 L188.0 150.3 L190.0 150.1 L192.0 150.0 L194.0 149.8 L196.0 149.5 L198.0 149.3 L200.0 149.1 L202.0 148.9 L204.0 148.8 L206.0 148.6 L208.0 148.4 L210.0 148.3 L212.0 148.2 L214.0 148.1 L216.0 148.0 L218.0 147.9 L220.0 147.9 L222.0 147.8 L224.0 147.8 L226.0 147.7 L228.0 147.7 L230.0 147.6 L232.0 147.6 L234.0 147.5 L236.0 147.4 L238.0 147.3 L240.0 147.1 L242.0 147.0 L244.0 146.8 L246.0 146.6 L248.0 146.3 L250.0 146.0 L252.0 145.7 L254.0 145.4 L256.0 145.1 L258.0 144.7 L260.0 144.3 L262.0 143.9 L264.0 143.5 L266.0 143.1 L268.0 142.8 L270.0 142.4 L272.0 142.1 L274.0 141.7 L276.0 141.5 L278.0 141.2 L280.0 141.0 L282.0 140.8 L284.0 140.7 L286.0 140.7 L288.0 140.6 L290.0 140.7 L292.0 140.7 L294.0 140.8"/>
                  <path class="trace-muted" d="M54.0 198.0 L56.0 195.9 L58.0 197.8 L60.0 199.8 L62.0 198.3 L64.0 196.6 L66.0 197.6 L68.0 199.0 L70.0 198.3 L72.0 197.4 L74.0 197.7 L76.0 198.4 L78.0 198.2 L80.0 197.7 L82.0 197.6 L84.0 198.5 L86.0 198.6 L88.0 197.3 L90.0 197.0 L92.0 198.8 L94.0 199.5 L96.0 197.2 L98.0 196.2 L100.0 198.7 L102.0 200.0 L104.0 197.5 L106.0 196.1 L108.0 198.3 L110.0 199.7 L112.0 197.9 L114.0 196.7 L116.0 197.9 L118.0 198.8 L120.0 198.1 L122.0 197.5 L124.0 197.9 L126.0 198.4 L128.0 198.2 L130.0 197.5 L132.0 197.7 L134.0 198.7 L136.0 198.6 L138.0 197.0 L140.0 197.1 L142.0 199.2 L144.0 199.3 L146.0 196.7 L148.0 196.4 L150.0 199.2 L152.0 199.7 L154.0 197.0 L156.0 196.3 L158.0 198.7 L160.0 199.4 L162.0 197.6 L164.0 197.0 L166.0 198.2 L168.0 198.7 L170.0 197.9 L172.0 197.6 L174.0 198.0 L176.0 198.4 L178.0 198.1 L180.0 197.3 L182.0 197.8 L184.0 199.0 L186.0 198.4 L188.0 196.6 L190.0 197.3 L192.0 199.7 L194.0 199.0 L196.0 196.3 L198.0 196.8 L200.0 199.6 L202.0 199.3 L204.0 196.7 L206.0 196.8 L208.0 198.9 L210.0 199.0 L212.0 197.4 L214.0 197.3 L216.0 198.3 L218.0 198.5 L220.0 197.8 L222.0 197.6 L224.0 198.1 L226.0 198.5 L228.0 197.9 L230.0 197.2 L232.0 198.1 L234.0 199.3 L236.0 198.1 L238.0 196.3 L240.0 197.7 L242.0 199.9 L244.0 198.5 L246.0 196.0 L248.0 197.3 L250.0 199.8 L252.0 198.8 L254.0 196.5 L256.0 197.2 L258.0 199.0 L260.0 198.7 L262.0 197.4 L264.0 197.5 L266.0 198.4 L268.0 198.3 L270.0 197.8 L272.0 197.6 L274.0 198.3 L276.0 198.6 L278.0 197.7 L280.0 197.0 L282.0 198.4 L284.0 199.4 L286.0 197.7 L288.0 196.2 L290.0 198.2 L292.0 200.1 L294.0 198.0"/>
                </g>
                <rect x="292" y="106" width="68" height="36" rx="8" class="stage-model-box"/>
                <text x="326" y="128" text-anchor="middle" class="stage-model-txt bd-mono">SleepFM</text>
                <text x="374" y="30" class="stage-out-head bd-mono">AASM STAGE</text>
                <g class="stage-bars">
                  <text x="374" y="54" class="bar-lbl-hi">Stage N2</text>
                  <text x="524" y="54" text-anchor="end" class="bar-val-hi bd-mono">0.88</text>
                  <rect x="374" y="61" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="61" width="132" height="8" rx="4" class="bar-fill-hi"/>

                  <text x="374" y="96" class="bar-lbl">Stage N3</text>
                  <text x="524" y="96" text-anchor="end" class="bar-val bd-mono">0.06</text>
                  <rect x="374" y="103" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="103" width="12" height="8" rx="4" class="bar-fill"/>

                  <text x="374" y="138" class="bar-lbl">Stage N1</text>
                  <text x="524" y="138" text-anchor="end" class="bar-val bd-mono">0.03</text>
                  <rect x="374" y="145" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="145" width="8" height="8" rx="4" class="bar-fill"/>

                  <text x="374" y="180" class="bar-lbl">REM</text>
                  <text x="524" y="180" text-anchor="end" class="bar-val bd-mono">0.02</text>
                  <rect x="374" y="187" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="187" width="7" height="8" rx="4" class="bar-fill"/>

                  <text x="374" y="220" class="bar-lbl">Wake</text>
                  <text x="524" y="220" text-anchor="end" class="bar-val bd-mono">0.01</text>
                  <rect x="374" y="227" width="150" height="7" rx="3.5" class="bar-track"/>
                  <rect x="374" y="227" width="6" height="7" rx="3.5" class="bar-fill"/>
                </g>
              </svg>
            </div>

            <div class="stage-panel stage-panel-szcore">
              <svg class="stage-svg" viewBox="0 0 540 248" role="img" aria-label="Focal left temporal rhythmic spike-and-wave onset scored with SeizureTransformer and SzCORE event rules">
                <rect x="0.5" y="0.5" width="539" height="247" rx="10" class="stage-canvas"/>
                <g class="stage-ch-labels bd-mono">
                  <text x="8" y="40">FP1-F7</text>
                  <text x="8" y="84">F7-T7</text>
                  <text x="8" y="128">T7-P7</text>
                  <text x="8" y="172">F3-C3</text>
                  <text x="8" y="216">C3-P3</text>
                </g>
                <g class="stage-rays" fill="none">
                  <path d="M254 36 C278 36, 286 124, 300 124"/>
                  <path d="M254 80 C276 80, 286 124, 300 124"/>
                  <path d="M254 124 L300 124"/>
                  <path d="M254 168 C276 168, 286 124, 300 124"/>
                  <path d="M254 212 C278 212, 286 124, 300 124"/>
                </g>
                <rect x="122" y="14" width="132" height="214" rx="6" class="stage-epoch"/>
                <text x="188" y="241" text-anchor="middle" class="stage-epoch-lbl bd-mono">ICTAL ONSET · SZCORE</text>
                <g fill="none" stroke-linecap="round" stroke-linejoin="round">
                  <path class="trace-muted" d="M54.0 34.6 L56.0 34.1 L58.0 34.0 L60.0 34.2 L62.0 34.6 L64.0 34.8 L66.0 34.8 L68.0 34.5 L70.0 34.2 L72.0 34.0 L74.0 34.1 L76.0 34.7 L78.0 35.7 L80.0 37.0 L82.0 38.3 L84.0 39.4 L86.0 40.0 L88.0 39.9 L90.0 39.1 L92.0 38.0 L94.0 36.6 L96.0 35.4 L98.0 34.5 L100.0 34.0 L102.0 34.0 L104.0 34.3 L106.0 34.6 L108.0 34.8 L110.0 34.7 L112.0 34.5 L114.0 34.2 L116.0 34.0 L118.0 34.2 L120.0 34.8 L122.0 35.0 L124.0 33.4 L126.0 32.8 L128.0 39.5 L130.0 40.7 L132.0 41.5 L134.0 41.8 L136.0 41.8 L138.0 41.4 L140.0 40.5 L142.0 39.3 L144.0 37.9 L146.0 37.2 L148.0 30.3 L150.0 37.7 L152.0 38.9 L154.0 38.9 L156.0 38.7 L158.0 38.3 L160.0 37.7 L162.0 37.0 L164.0 36.2 L166.0 35.3 L168.0 34.6 L170.0 32.3 L172.0 33.0 L174.0 39.0 L176.0 40.2 L178.0 41.1 L180.0 41.5 L182.0 41.6 L184.0 41.2 L186.0 40.5 L188.0 39.5 L190.0 38.2 L192.0 37.5 L194.0 30.8 L196.0 38.6 L198.0 39.4 L200.0 39.4 L202.0 39.1 L204.0 38.6 L206.0 37.9 L208.0 37.0 L210.0 36.0 L212.0 35.0 L214.0 34.4 L216.0 31.3 L218.0 33.3 L220.0 38.6 L222.0 39.7 L224.0 40.6 L226.0 41.1 L228.0 41.2 L230.0 41.0 L232.0 40.4 L234.0 39.5 L236.0 38.3 L238.0 37.6 L240.0 31.3 L242.0 39.3 L244.0 40.0 L246.0 40.0 L248.0 39.6 L250.0 39.0 L252.0 38.1 L254.0 37.1 L256.0 36.0 L258.0 34.9 L260.0 34.2 L262.0 30.3 L264.0 33.6 L266.0 38.2 L268.0 39.2 L270.0 40.1 L272.0 40.6 L274.0 40.8 L276.0 40.6 L278.0 40.1 L280.0 39.3 L282.0 38.3 L284.0 37.6 L286.0 32.0 L288.0 39.9 L290.0 40.5 L292.0 40.5 L294.0 40.1"/>
                  <path class="trace-hi" d="M54.0 76.7 L56.0 76.1 L58.0 76.1 L60.0 76.8 L62.0 78.0 L64.0 79.4 L66.0 80.8 L68.0 81.8 L70.0 82.4 L72.0 82.5 L74.0 82.2 L76.0 81.7 L78.0 81.3 L80.0 81.1 L82.0 81.1 L84.0 81.3 L86.0 81.4 L88.0 81.4 L90.0 80.9 L92.0 80.0 L94.0 78.8 L96.0 77.6 L98.0 76.6 L100.0 76.1 L102.0 76.2 L104.0 77.0 L106.0 78.2 L108.0 79.7 L110.0 81.0 L112.0 82.0 L114.0 82.5 L116.0 82.4 L118.0 82.1 L120.0 81.6 L122.0 80.9 L124.0 77.6 L126.0 75.8 L128.0 87.3 L130.0 88.7 L132.0 89.3 L134.0 89.3 L136.0 88.5 L138.0 87.2 L140.0 85.3 L142.0 83.0 L144.0 80.5 L146.0 79.4 L148.0 67.4 L150.0 81.5 L152.0 84.3 L154.0 85.1 L156.0 85.5 L158.0 85.5 L160.0 85.1 L162.0 84.3 L164.0 83.2 L166.0 81.8 L168.0 80.5 L170.0 76.1 L172.0 76.9 L174.0 87.2 L176.0 88.6 L178.0 89.4 L180.0 89.4 L182.0 88.8 L184.0 87.5 L186.0 85.7 L188.0 83.4 L190.0 80.9 L192.0 79.7 L194.0 67.8 L196.0 82.4 L198.0 84.6 L200.0 85.2 L202.0 85.5 L204.0 85.3 L206.0 84.8 L208.0 83.8 L210.0 82.6 L212.0 81.2 L214.0 80.1 L216.0 74.4 L218.0 77.9 L220.0 87.0 L222.0 88.5 L224.0 89.3 L226.0 89.5 L228.0 88.9 L230.0 87.7 L232.0 86.0 L234.0 83.8 L236.0 81.3 L238.0 79.9 L240.0 68.5 L242.0 83.2 L244.0 85.0 L246.0 85.5 L248.0 85.6 L250.0 85.3 L252.0 84.5 L254.0 83.4 L256.0 82.1 L258.0 80.6 L260.0 79.7 L262.0 72.7 L264.0 78.7 L266.0 86.7 L268.0 88.2 L270.0 89.1 L272.0 89.3 L274.0 88.9 L276.0 87.8 L278.0 86.1 L280.0 84.0 L282.0 81.6 L284.0 80.0 L286.0 69.6 L288.0 84.1 L290.0 85.5 L292.0 85.9 L294.0 85.8"/>
                  <path class="trace-hi" d="M54.0 123.2 L56.0 123.1 L58.0 123.2 L60.0 123.5 L62.0 124.2 L64.0 125.1 L66.0 126.2 L68.0 127.2 L70.0 127.8 L72.0 127.7 L74.0 127.0 L76.0 125.8 L78.0 124.3 L80.0 122.9 L82.0 121.7 L84.0 121.1 L86.0 121.1 L88.0 121.5 L90.0 122.1 L92.0 122.7 L94.0 123.1 L96.0 123.3 L98.0 123.2 L100.0 123.1 L102.0 123.2 L104.0 123.6 L106.0 124.3 L108.0 125.3 L110.0 126.4 L112.0 127.3 L114.0 127.8 L116.0 127.6 L118.0 126.8 L120.0 125.5 L122.0 126.0 L124.0 122.2 L126.0 119.9 L128.0 130.9 L130.0 131.8 L132.0 132.0 L134.0 131.7 L136.0 130.7 L138.0 129.3 L140.0 127.4 L142.0 125.3 L144.0 123.1 L146.0 122.3 L148.0 110.8 L150.0 125.3 L152.0 128.6 L154.0 129.9 L156.0 130.8 L158.0 131.1 L160.0 130.9 L162.0 130.2 L164.0 129.1 L166.0 127.5 L168.0 126.0 L170.0 121.2 L172.0 121.5 L174.0 131.3 L176.0 132.3 L178.0 132.5 L180.0 132.2 L182.0 131.2 L184.0 129.7 L186.0 127.8 L188.0 125.5 L190.0 123.2 L192.0 122.2 L194.0 110.7 L196.0 125.7 L198.0 128.4 L200.0 129.6 L202.0 130.3 L204.0 130.6 L206.0 130.4 L208.0 129.6 L210.0 128.5 L212.0 127.0 L214.0 125.8 L216.0 119.9 L218.0 122.9 L220.0 131.6 L222.0 132.6 L224.0 132.9 L226.0 132.6 L228.0 131.7 L230.0 130.1 L232.0 128.1 L234.0 125.8 L236.0 123.4 L238.0 122.1 L240.0 111.1 L242.0 126.2 L244.0 128.4 L246.0 129.4 L248.0 130.0 L250.0 130.1 L252.0 129.8 L254.0 129.0 L256.0 127.9 L258.0 126.5 L260.0 125.6 L262.0 118.5 L264.0 124.2 L266.0 131.7 L268.0 132.8 L270.0 133.2 L272.0 133.0 L274.0 132.1 L276.0 130.5 L278.0 128.6 L280.0 126.2 L282.0 123.7 L284.0 122.1 L286.0 111.9 L288.0 126.6 L290.0 128.4 L292.0 129.2 L294.0 129.7"/>
                  <path class="trace-muted" d="M54.0 171.2 L56.0 171.3 L58.0 170.9 L60.0 170.2 L62.0 169.4 L64.0 168.7 L66.0 168.3 L68.0 168.2 L70.0 168.3 L72.0 168.3 L74.0 168.1 L76.0 167.5 L78.0 166.7 L80.0 165.8 L82.0 164.9 L84.0 164.5 L86.0 164.5 L88.0 165.2 L90.0 166.5 L92.0 168.0 L94.0 169.5 L96.0 170.6 L98.0 171.2 L100.0 171.3 L102.0 170.8 L104.0 170.0 L106.0 169.2 L108.0 168.6 L110.0 168.3 L112.0 168.2 L114.0 168.3 L116.0 168.2 L118.0 168.0 L120.0 167.4 L122.0 169.2 L124.0 167.8 L126.0 166.8 L128.0 169.1 L130.0 168.9 L132.0 168.6 L134.0 168.3 L136.0 167.9 L138.0 167.5 L140.0 167.2 L142.0 166.9 L144.0 166.6 L146.0 166.8 L148.0 164.4 L150.0 168.5 L152.0 169.8 L154.0 170.5 L156.0 171.1 L158.0 171.4 L160.0 171.5 L162.0 171.3 L164.0 170.9 L166.0 170.3 L168.0 169.6 L170.0 168.0 L172.0 167.7 L174.0 169.7 L176.0 169.4 L178.0 169.1 L180.0 168.7 L182.0 168.2 L184.0 167.7 L186.0 167.2 L188.0 166.7 L190.0 166.3 L192.0 166.4 L194.0 163.9 L196.0 168.1 L198.0 169.3 L200.0 170.0 L202.0 170.6 L204.0 171.0 L206.0 171.2 L208.0 171.2 L210.0 170.9 L212.0 170.4 L214.0 169.9 L216.0 168.1 L218.0 168.5 L220.0 170.2 L222.0 170.0 L224.0 169.6 L226.0 169.1 L228.0 168.5 L230.0 167.9 L232.0 167.3 L234.0 166.7 L236.0 166.2 L238.0 166.1 L240.0 163.6 L242.0 167.8 L244.0 168.8 L246.0 169.5 L248.0 170.1 L250.0 170.6 L252.0 170.8 L254.0 170.9 L256.0 170.7 L258.0 170.4 L260.0 170.1 L262.0 168.1 L264.0 169.2 L266.0 170.7 L268.0 170.5 L270.0 170.1 L272.0 169.6 L274.0 169.0 L276.0 168.3 L278.0 167.5 L280.0 166.8 L282.0 166.2 L284.0 165.9 L286.0 163.5 L288.0 167.5 L290.0 168.4 L292.0 169.0 L294.0 169.6"/>
                  <path class="trace-muted" d="M54.0 215.2 L56.0 215.2 L58.0 214.6 L60.0 213.5 L62.0 212.0 L64.0 210.5 L66.0 209.2 L68.0 208.5 L70.0 208.4 L72.0 208.9 L74.0 209.7 L76.0 210.7 L78.0 211.6 L80.0 212.1 L82.0 212.3 L84.0 212.3 L86.0 212.3 L88.0 212.4 L90.0 212.7 L92.0 213.4 L94.0 214.1 L96.0 214.9 L98.0 215.3 L100.0 215.1 L102.0 214.4 L104.0 213.2 L106.0 211.7 L108.0 210.2 L110.0 209.1 L112.0 208.4 L114.0 208.5 L116.0 209.0 L118.0 209.9 L120.0 210.9 L122.0 211.3 L124.0 209.9 L126.0 209.0 L128.0 211.5 L130.0 211.7 L132.0 211.8 L134.0 212.0 L136.0 212.1 L138.0 212.2 L140.0 212.3 L142.0 212.4 L144.0 212.4 L146.0 212.7 L148.0 210.3 L150.0 214.3 L152.0 215.3 L154.0 215.7 L156.0 215.9 L158.0 215.7 L160.0 215.3 L162.0 214.7 L164.0 213.8 L166.0 212.8 L168.0 211.9 L170.0 210.1 L172.0 209.8 L174.0 211.9 L176.0 211.9 L178.0 211.9 L180.0 211.9 L182.0 211.9 L184.0 211.9 L186.0 211.8 L188.0 211.8 L190.0 211.8 L192.0 212.2 L194.0 209.8 L196.0 214.0 L198.0 215.1 L200.0 215.6 L202.0 215.8 L204.0 215.8 L206.0 215.5 L208.0 215.0 L210.0 214.2 L212.0 213.3 L214.0 212.4 L216.0 210.4 L218.0 210.6 L220.0 212.3 L222.0 212.2 L224.0 212.0 L226.0 211.9 L228.0 211.7 L230.0 211.6 L232.0 211.5 L234.0 211.4 L236.0 211.3 L238.0 211.6 L240.0 209.4 L242.0 213.7 L244.0 214.7 L246.0 215.3 L248.0 215.7 L250.0 215.8 L252.0 215.6 L254.0 215.2 L256.0 214.5 L258.0 213.7 L260.0 212.9 L262.0 210.6 L264.0 211.4 L266.0 212.8 L268.0 212.6 L270.0 212.3 L272.0 212.1 L274.0 211.8 L276.0 211.5 L278.0 211.2 L280.0 211.0 L282.0 210.8 L284.0 211.0 L286.0 209.0 L288.0 213.3 L290.0 214.3 L292.0 214.9 L294.0 215.4"/>
                </g>
                <rect x="288" y="106" width="76" height="36" rx="8" class="stage-model-box"/>
                <text x="326" y="128" text-anchor="middle" class="stage-model-txt bd-mono">SeizureTF</text>
                <text x="374" y="32" class="stage-out-head bd-mono">SZCORE EVENT</text>
                <g class="stage-bars">
                  <text x="374" y="60" class="bar-lbl-hi">Ictal Score</text>
                  <text x="524" y="60" text-anchor="end" class="bar-val-hi bd-mono">0.94</text>
                  <rect x="374" y="68" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="68" width="141" height="8" rx="4" class="bar-fill-hi"/>

                  <text x="374" y="108" class="bar-lbl">Background</text>
                  <text x="524" y="108" text-anchor="end" class="bar-val bd-mono">0.06</text>
                  <rect x="374" y="116" width="150" height="8" rx="4" class="bar-track"/>
                  <rect x="374" y="116" width="12" height="8" rx="4" class="bar-fill"/>

                  <text x="374" y="160" class="bar-lbl">Event Sensitivity</text>
                  <text x="524" y="160" text-anchor="end" class="bar-val-good bd-mono">3 / 3</text>

                  <text x="374" y="188" class="bar-lbl">False Alarms / h</text>
                  <text x="524" y="188" text-anchor="end" class="bar-val-good bd-mono">0.00</text>

                  <text x="374" y="216" class="bar-lbl">Causal Window</text>
                  <text x="524" y="216" text-anchor="end" class="bar-val bd-mono">60.0 s</text>
                </g>
              </svg>
            </div>
          </div>
        </div>

        <div class="inst-strip">
          <span class="inst-label">Maintained across</span>
          <div class="inst-logos">
            <img src="_static/institution_logos/ucsd.png" alt="UC San Diego" width="128" height="24" loading="lazy" decoding="async"/>
            <img src="_static/institution_logos/inria.png" alt="Inria" width="63" height="22" loading="lazy" decoding="async"/>
            <img src="_static/institution_logos/unifreiburg.png" alt="Uni Freiburg" width="29" height="34" loading="lazy" decoding="async"/>
            <img src="_static/institution_logos/donders.svg" alt="Donders Institute" width="78" height="26" loading="lazy" decoding="async"/>
            <img src="_static/institution_logos/yneuro.png" alt="Yneuro" width="57" height="24" loading="lazy" decoding="async"/>
          </div>
          <a class="inst-contrib bd-mono" href="https://github.com/braindecode/braindecode/graphs/contributors" target="_blank" rel="noopener">+ 133 community contributors</a>
        </div>
      </div>
    </section>

    <section class="foundation-band" id="foundation-hub">
      <div class="container foundation-grid">
        <div class="foundation-copy">
          <h2 class="foundation-title">48 foundation models, one from_pretrained() call.</h2>
          <p class="foundation-desc">
            Deterministic safetensors checkpoints with auditable weight provenance, spherical-spline ChannelLayer adaptation across arbitrary sensor montages, and single-graph torch.compile execution.
          </p>
          <div class="foundation-pills bd-mono">
            <a class="fb-pill warm" href="generated/braindecode.models.EEGPT.html">EEGPT</a>
            <a class="fb-pill warm" href="generated/braindecode.models.Labram.html">LaBraM</a>
            <a class="fb-pill warm" href="generated/braindecode.models.REVE.html">REVE</a>
            <a class="fb-pill warm" href="generated/braindecode.models.CBraMod.html">CBraMod</a>
            <a class="fb-pill warm" href="generated/braindecode.models.SAMBA.html">SAMBA</a>
            <a class="fb-pill" href="generated/braindecode.models.SignalJEPA.html">SignalJEPA</a>
            <a class="fb-pill" href="generated/braindecode.models.SleepFM.html">SleepFM</a>
            <a class="fb-pill" href="generated/braindecode.models.MEGXL.html">MEGXL</a>
            <a class="fb-pill" href="generated/braindecode.models.GeoNIRS.html">GeoNIRS</a>
            <a class="fb-pill" href="generated/braindecode.models.TinyMyo.html">TinyMyo</a>
          </div>
          <div class="foundation-links">
            <a class="btn btn-copper" href="https://huggingface.co/braindecode" target="_blank" rel="noopener">Browse 140+ Hub repos <span aria-hidden="true">→</span></a>
            <a class="fb-link" href="user_guide/channel_strategies.html">Channel strategies</a>
            <a class="fb-link" href="models/categorization/lbm.html">Foundation catalog</a>
          </div>
        </div>

        <div class="repl-card">
          <div class="repl-top bd-mono">
            <span>foundation_decode.py · safetensors + ChannelLayer</span>
            <span class="copy-wrap">
              <button type="button" class="code-copy-btn" data-copy="from braindecode.models import EEGPT&#10;&#10;model = EEGPT.from_pretrained(&#10;    &quot;braindecode/eegpt-pretrained&quot;,&#10;    chs_info=raw.info[&quot;chs&quot;],&#10;    channel_strategy=&quot;spline&quot;,&#10;    n_outputs=4,&#10;)&#10;probs = model(X_test).softmax(dim=-1)" aria-label="Copy EEGPT from_pretrained Python snippet">Copy snippet</button>
              <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
            </span>
          </div>
          <pre class="repl-code bd-mono" tabindex="0"><code><span class="cl"><span class="tok-kw">from</span> braindecode.models <span class="tok-kw">import</span> <span class="tok-cls">EEGPT</span></span><span class="cl"></span><span class="cl">model = <span class="tok-cls">EEGPT</span>.<span class="tok-fn">from_pretrained</span>(</span><span class="cl">    <span class="tok-str">"braindecode/eegpt-pretrained"</span>,</span><span class="cl">    <span class="tok-attr">chs_info</span>=raw.info[<span class="tok-str">"chs"</span>],</span><span class="cl">    <span class="tok-attr">channel_strategy</span>=<span class="tok-str">"spline"</span>,</span><span class="cl">    <span class="tok-attr">n_outputs</span>=<span class="tok-num">4</span>,</span><span class="cl">)</span><span class="cl">probs = model(X_test).<span class="tok-fn">softmax</span>(<span class="tok-attr">dim</span>=-<span class="tok-num">1</span>)  <span class="tok-com"># Any montage in</span></span></code></pre>
          <div class="repl-out bd-mono">
            <span>&gt;&gt;&gt; probs[0]  -&gt;  tensor([0.03, 0.92, 0.03, 0.02])</span>
            <span class="repl-verified">safetensors · spline montage</span>
          </div>
        </div>
      </div>
    </section>

    <section class="section section-alt">
      <div class="container">
        <div class="sec-head">
          <h2 class="section-title">One tensor interface across every neurophysiological signal.</h2>
          <p class="section-sub">Pass raw MNE objects, BIDS directories, or Zarr Hub streams; spatial filters and tokenizers size themselves from the recording metadata.</p>
        </div>

        <div class="ruled-bento">
          <div class="rb-cell rb-span-7">
            <div>
              <div class="rb-top">
                <h3 class="rb-title">Scalp &amp; Clinical EEG</h3>
                <span class="rb-stat">72 models</span>
              </div>
              <p class="rb-desc">19–128 channel montages at 128–512 Hz across motor imagery, ERP, clinical pathology, sleep staging, and universal representation learning — backed by 150+ MOABB and 700+ EEGDash datasets.</p>
            </div>
            <div class="rb-tags bd-mono">
              <a class="rb-tag hi" href="generated/braindecode.models.EEGNeX.html">EEGNeX</a>
              <a class="rb-tag hi" href="generated/braindecode.models.EEGConformer.html">EEGConformer</a>
              <a class="rb-tag warm" href="generated/braindecode.models.Labram.html">LaBraM</a>
              <a class="rb-tag warm" href="generated/braindecode.models.EEGPT.html">EEGPT</a>
              <a class="rb-tag warm" href="generated/braindecode.models.REVE.html">REVE</a>
              <a class="rb-tag warm" href="generated/braindecode.models.SAMBA.html">SAMBA</a>
              <a class="rb-tag warm" href="generated/braindecode.models.Pratibha2.html">Pratibha2</a>
              <a class="rb-tag" href="generated/braindecode.models.ShallowFBCSPNet.html">ShallowFBCSPNet</a>
              <a class="rb-tag" href="generated/braindecode.models.SeizureTransformer.html">SeizureTransformer</a>
            </div>
          </div>

          <div class="rb-cell rb-span-5">
            <div>
              <div class="rb-top">
                <h3 class="rb-title">Intracranial iEEG, ECoG &amp; sEEG</h3>
                <span class="rb-stat sage">9 models</span>
              </div>
              <p class="rb-desc">High-gamma (500–2048 Hz) cortical grids and stereotactic shafts with 3D parcel coordinates for speech decoding and epileptogenic localization.</p>
            </div>
            <div class="rb-tags bd-mono">
              <a class="rb-tag warm" href="generated/braindecode.models.DuIN.html">DuIN</a>
              <a class="rb-tag warm" href="generated/braindecode.models.BrainBERT.html">BrainBERT</a>
              <a class="rb-tag warm" href="generated/braindecode.models.PopulationTransformer.html">PopulationTransformer</a>
              <a class="rb-tag warm" href="generated/braindecode.models.CORTEG.html">CORTEG</a>
              <a class="rb-tag warm" href="generated/braindecode.models.MAPA.html">MAPA</a>
              <a class="rb-tag warm" href="generated/braindecode.models.Brant.html">Brant</a>
              <a class="rb-tag warm" href="generated/braindecode.models.DIVER1.html">DIVER1</a>
            </div>
          </div>

          <div class="rb-cell rb-span-4">
            <div>
              <div class="rb-top">
                <h3 class="rb-title">fNIRS Hemodynamics</h3>
                <span class="rb-stat warm">8 models</span>
              </div>
              <p class="rb-desc">Paired HbO/HbR optical optode arrays (5–10 Hz) with source-detector geometry encoders and diagonal state-space (S4D) kernels.</p>
            </div>
            <div class="rb-tags bd-mono">
              <a class="rb-tag warm" href="generated/braindecode.models.GeoNIRS.html">GeoNIRS</a>
              <a class="rb-tag warm" href="generated/braindecode.models.MtReNIRS.html">MtReNIRS</a>
              <a class="rb-tag warm" href="generated/braindecode.models.JungMAE.html">JungMAE</a>
              <a class="rb-tag" href="generated/braindecode.models.FNIRS4D.html">FNIRS4D</a>
              <a class="rb-tag" href="generated/braindecode.models.FNIRSNet.html">FNIRSNet</a>
            </div>
          </div>

          <div class="rb-cell rb-span-4">
            <div>
              <div class="rb-top">
                <h3 class="rb-title">Whole-Head MEG</h3>
                <span class="rb-stat">7 models</span>
              </div>
              <p class="rb-desc">306-sensor SQUID and wearable OPM arrays with residual vector quantizers, source-space GPT, and visual stimulus decoding.</p>
            </div>
            <div class="rb-tags bd-mono">
              <a class="rb-tag warm" href="generated/braindecode.models.MEGXL.html">MEGXL</a>
              <a class="rb-tag warm" href="generated/braindecode.models.MEGGPT.html">MEGGPT</a>
              <a class="rb-tag warm" href="generated/braindecode.models.BBL.html">BBL</a>
              <a class="rb-tag hi" href="generated/braindecode.models.BrainOmni.html">BrainOmni</a>
              <a class="rb-tag" href="generated/braindecode.models.NICE.html">NICE</a>
            </div>
          </div>

          <div class="rb-cell rb-span-4">
            <div>
              <div class="rb-top">
                <h3 class="rb-title">Neuromotor sEMG</h3>
                <span class="rb-stat warm">7 models</span>
              </div>
              <p class="rb-desc">2 kHz surface motor-unit potentials for zero-calibration CTC touch typing, handwriting, gesture classification, and 20-DoF hand-pose tracking.</p>
            </div>
            <div class="rb-tags bd-mono">
              <a class="rb-tag warm" href="generated/braindecode.models.TinyMyo.html">TinyMyo</a>
              <a class="rb-tag hi" href="generated/braindecode.models.MetaNeuromotorHand.html">MetaNeuromotorHand</a>
              <a class="rb-tag" href="generated/braindecode.models.VEMG2Pose.html">VEMG2Pose</a>
              <a class="rb-tag" href="generated/braindecode.models.EMG2QwertyNet.html">EMG2QwertyNet</a>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section wf-section">
      <div class="container">
        <div class="sec-head">
          <h2 class="section-title">From raw recording to evaluated model in twelve lines.</h2>
          <p class="section-sub">Swap datasets, channel montages, and training loops without rewriting model constructors.</p>
        </div>

        <div class="wf-stage">
          <fieldset class="wf-tabs" aria-label="Choose end-to-end Python workflow">
            <label class="wf-tab">
              <input type="radio" name="bd-wf" value="moabb" class="sr-only" checked/>
              <span>Supervised Benchmark (MOABB + EEGNeX)</span>
            </label>
            <label class="wf-tab">
              <input type="radio" name="bd-wf" value="montage" class="sr-only"/>
              <span>Mixed-Montage Fine-Tuning (LaBraM + ChannelLayer)</span>
            </label>
            <label class="wf-tab">
              <input type="radio" name="bd-wf" value="szcore" class="sr-only"/>
              <span>Clinical Seizure Scoring (SeizureTransformer + SzCORE)</span>
            </label>
          </fieldset>

          <div class="wf-pane wf-pane-moabb">
            <div class="wf-code-card">
              <div class="repl-top bd-mono">
                <span>benchmark_moabb.py</span>
                <span class="copy-wrap">
                  <button type="button" class="code-copy-btn" data-copy="from braindecode import EEGClassifier&#10;from braindecode.datasets import MOABBDataset&#10;from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess&#10;&#10;dataset = MOABBDataset(&quot;BNCI2014_001&quot;, subject_ids=[3])&#10;preprocess(dataset, [&#10;    Preprocessor(&quot;pick_types&quot;, eeg=True),&#10;    Preprocessor(&quot;filter&quot;, l_freq=4.0, h_freq=38.0),&#10;])&#10;windows = create_windows_from_events(dataset, preload=True)&#10;clf = EEGClassifier(&quot;EEGNeX&quot;, max_epochs=20, batch_size=64)&#10;clf.fit(windows, y=None)" aria-label="Copy Supervised Benchmark MOABB workflow">Copy</button>
                  <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
                </span>
              </div>
              <pre class="repl-code bd-mono" tabindex="0"><code><span class="cl"><span class="tok-kw">from</span> braindecode <span class="tok-kw">import</span> <span class="tok-cls">EEGClassifier</span></span><span class="cl"><span class="tok-kw">from</span> braindecode.datasets <span class="tok-kw">import</span> <span class="tok-cls">MOABBDataset</span></span><span class="cl"><span class="tok-kw">from</span> braindecode.preprocessing <span class="tok-kw">import</span> <span class="tok-cls">Preprocessor</span>, <span class="tok-fn">create_windows_from_events</span>, <span class="tok-fn">preprocess</span></span><span class="cl"></span><span class="cl">dataset = <span class="tok-cls">MOABBDataset</span>(<span class="tok-str">"BNCI2014_001"</span>, <span class="tok-attr">subject_ids</span>=[<span class="tok-num">3</span>])</span><span class="cl"><span class="tok-fn">preprocess</span>(dataset, [</span><span class="cl">    <span class="tok-cls">Preprocessor</span>(<span class="tok-str">"pick_types"</span>, <span class="tok-attr">eeg</span>=<span class="tok-kw">True</span>),</span><span class="cl">    <span class="tok-cls">Preprocessor</span>(<span class="tok-str">"filter"</span>, <span class="tok-attr">l_freq</span>=<span class="tok-num">4.0</span>, <span class="tok-attr">h_freq</span>=<span class="tok-num">38.0</span>),</span><span class="cl">])</span><span class="cl">windows = <span class="tok-fn">create_windows_from_events</span>(dataset, <span class="tok-attr">preload</span>=<span class="tok-kw">True</span>)</span><span class="cl">clf = <span class="tok-cls">EEGClassifier</span>(<span class="tok-str">"EEGNeX"</span>, <span class="tok-attr">max_epochs</span>=<span class="tok-num">20</span>, <span class="tok-attr">batch_size</span>=<span class="tok-num">64</span>)</span><span class="cl">clf.<span class="tok-fn">fit</span>(windows, <span class="tok-attr">y</span>=<span class="tok-kw">None</span>)</span></code></pre>
            </div>
            <div class="wf-meta bd-pane">
              <h3 class="wf-meta-title">Tensor shape trace</h3>
              <dl class="wf-shapes bd-mono">
                <div><dt>MOABB Raw</dt><dd>(22 ch, 672528) @ 250 Hz</dd></div>
                <div><dt>WindowsDataset</dt><dd>(288, 22, 1000) float32</dd></div>
                <div><dt>EEGNeX</dt><dd>55.9 K params · derived shapes</dd></div>
                <div><dt>clf.predict_proba</dt><dd>(288, 4) calibrated probs</dd></div>
              </dl>
              <a class="wf-tut-link" href="auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html">Open trialwise MOABB tutorial <span aria-hidden="true">→</span></a>
            </div>
          </div>

          <div class="wf-pane wf-pane-montage">
            <div class="wf-code-card">
              <div class="repl-top bd-mono">
                <span>finetune_montage.py</span>
                <span class="copy-wrap">
                  <button type="button" class="code-copy-btn" data-copy="from torch.utils.data import DataLoader&#10;from braindecode.datasets import pad_channels_collate&#10;from braindecode.models import Labram&#10;&#10;windows.set_return_ch_pos(True)&#10;loader = DataLoader(windows, batch_size=32, collate_fn=pad_channels_collate)&#10;model = Labram.from_pretrained(&#10;    &quot;braindecode/labram-pretrained&quot;,&#10;    chs_info=raw.info[&quot;chs&quot;],&#10;    channel_strategy=&quot;spline&quot;,&#10;    n_outputs=4,&#10;)&#10;X, y, _, pos, ch_mask = next(iter(loader))" aria-label="Copy Mixed-Montage Foundation Fine-Tuning workflow">Copy</button>
                  <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
                </span>
              </div>
              <pre class="repl-code bd-mono" tabindex="0"><code><span class="cl"><span class="tok-kw">from</span> torch.utils.data <span class="tok-kw">import</span> <span class="tok-cls">DataLoader</span></span><span class="cl"><span class="tok-kw">from</span> braindecode.datasets <span class="tok-kw">import</span> <span class="tok-fn">pad_channels_collate</span></span><span class="cl"><span class="tok-kw">from</span> braindecode.models <span class="tok-kw">import</span> <span class="tok-cls">Labram</span></span><span class="cl"></span><span class="cl">windows.<span class="tok-fn">set_return_ch_pos</span>(<span class="tok-kw">True</span>)</span><span class="cl">loader = <span class="tok-cls">DataLoader</span>(windows, <span class="tok-attr">batch_size</span>=<span class="tok-num">32</span>, <span class="tok-attr">collate_fn</span>=<span class="tok-fn">pad_channels_collate</span>)</span><span class="cl">model = <span class="tok-cls">Labram</span>.<span class="tok-fn">from_pretrained</span>(</span><span class="cl">    <span class="tok-str">"braindecode/labram-pretrained"</span>,</span><span class="cl">    <span class="tok-attr">chs_info</span>=raw.info[<span class="tok-str">"chs"</span>],</span><span class="cl">    <span class="tok-attr">channel_strategy</span>=<span class="tok-str">"spline"</span>,</span><span class="cl">    <span class="tok-attr">n_outputs</span>=<span class="tok-num">4</span>,</span><span class="cl">)</span><span class="cl">X, y, _, pos, ch_mask = <span class="tok-fn">next</span>(<span class="tok-fn">iter</span>(loader))</span></code></pre>
            </div>
            <div class="wf-meta bd-pane">
              <h3 class="wf-meta-title">Tensor shape trace</h3>
              <dl class="wf-shapes bd-mono">
                <div><dt>Heterogeneous caps</dt><dd>19-ch + 64-ch recordings</dd></div>
                <div><dt>Padded batch X</dt><dd>(32, 64, 800) + ch_mask</dd></div>
                <div><dt>ChannelLayer</dt><dd>Spherical spline -> 10-20</dd></div>
                <div><dt>LaBraM head</dt><dd>5.9 M params · (32, 4) logits</dd></div>
              </dl>
              <a class="wf-tut-link" href="auto_examples/model_building/plot_heterogeneous_montage_training.html">Open mixed-montage tutorial <span aria-hidden="true">→</span></a>
            </div>
          </div>

          <div class="wf-pane wf-pane-szcore">
            <div class="wf-code-card">
              <div class="repl-top bd-mono">
                <span>clinical_szcore.py</span>
                <span class="copy-wrap">
                  <button type="button" class="code-copy-btn" data-copy="import numpy as np&#10;import torch&#10;from braindecode.models import SeizureTransformer&#10;from braindecode.training import szcore_event_scores&#10;&#10;model = SeizureTransformer(n_chans=19, n_outputs=1, n_times=15360)&#10;model.eval()&#10;with torch.no_grad():&#10;    probs = model(X_record).sigmoid().squeeze(-1).numpy()&#10;alarms = window_starts[probs &gt; 0.85]&#10;events = np.c_[alarms, alarms + 60.0]&#10;scores = szcore_event_scores(ref_seizures, events, duration_s)" aria-label="Copy Clinical Seizure Event Scoring workflow">Copy</button>
                  <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
                </span>
              </div>
              <pre class="repl-code bd-mono" tabindex="0"><code><span class="cl"><span class="tok-kw">import</span> numpy <span class="tok-kw">as</span> np</span><span class="cl"><span class="tok-kw">import</span> torch</span><span class="cl"><span class="tok-kw">from</span> braindecode.models <span class="tok-kw">import</span> <span class="tok-cls">SeizureTransformer</span></span><span class="cl"><span class="tok-kw">from</span> braindecode.training <span class="tok-kw">import</span> <span class="tok-fn">szcore_event_scores</span></span><span class="cl"></span><span class="cl">model = <span class="tok-cls">SeizureTransformer</span>(<span class="tok-attr">n_chans</span>=<span class="tok-num">19</span>, <span class="tok-attr">n_outputs</span>=<span class="tok-num">1</span>, <span class="tok-attr">n_times</span>=<span class="tok-num">15360</span>)</span><span class="cl">model.<span class="tok-fn">eval</span>()</span><span class="cl"><span class="tok-kw">with</span> torch.<span class="tok-fn">no_grad</span>():</span><span class="cl">    probs = model(X_record).<span class="tok-fn">sigmoid</span>().<span class="tok-fn">squeeze</span>(-<span class="tok-num">1</span>).<span class="tok-fn">numpy</span>()</span><span class="cl">alarms = window_starts[probs &gt; <span class="tok-num">0.85</span>]</span><span class="cl">events = np.c_[alarms, alarms + <span class="tok-num">60.0</span>]</span><span class="cl">scores = <span class="tok-fn">szcore_event_scores</span>(ref_seizures, events, duration_s)</span></code></pre>
            </div>
            <div class="wf-meta bd-pane">
              <h3 class="wf-meta-title">Tensor shape trace</h3>
              <dl class="wf-shapes bd-mono">
                <div><dt>Bipolar EDF</dt><dd>(N, 19, 15360) @ 256 Hz</dd></div>
                <div><dt>SeizureTransformer</dt><dd>37.8 M params · 60 s context</dd></div>
                <div><dt>Alarm intervals</dt><dd>[onset_s, offset_s] pairs</dd></div>
                <div><dt>szcore_event_scores</dt><dd>tp: 3/3 · 0.00 FA/hour</dd></div>
              </dl>
              <a class="wf-tut-link" href="auto_examples/applied_examples/plot_seizure_alarm_chbmit.html">Open CHB-MIT SzCORE tutorial <span aria-hidden="true">→</span></a>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section section-alt zoo-section" id="model-zoo">
      <div class="container">
        <div class="sec-head">
          <h2 class="section-title">Explore published architectures by modality and backbone.</h2>
          <p class="section-sub">Every model inherits <code class="bd-mono">EEGModuleMixin</code> for automatic <code class="bd-mono">n_chans</code>, <code class="bd-mono">n_times</code>, <code class="bd-mono">sfreq</code>, and <code class="bd-mono">chs_info</code> shape derivation plus Hub serialization.</p>
        </div>

.. raw:: html
    :file: generated/zoo_browser.html

.. raw:: html

      </div>
    </section>

    <section class="section arch-section">
      <div class="container">
        <div class="sec-head">
          <h2 class="section-title">Composable PyTorch subpackages from raw files to attribution maps.</h2>
          <p class="section-sub">Use the full end-to-end pipeline or import individual functional kernels and layers into custom training loops.</p>
        </div>

        <div class="mod-ledger">
          <div class="mod-col">
            <h3 class="mod-col-title">Ingest, Clean &amp; Window</h3>
            <a class="mod-row" href="api.html#datasets">
              <span class="mod-name bd-mono">braindecode.datasets</span>
              <span class="mod-desc">MOABB, BIDS, Zarr Hub streaming, TUH, HBN, SleepPhysionet, and MNE wrappers.</span>
            </a>
            <a class="mod-row" href="api.html#preprocessing">
              <span class="mod-name bd-mono">braindecode.preprocessing</span>
              <span class="mod-desc">MNE-backed filtering, EEGPrep artifact cleaning, and event or fixed windowing.</span>
            </a>
            <a class="mod-row" href="api.html#samplers">
              <span class="mod-name bd-mono">braindecode.samplers</span>
              <span class="mod-desc">Balanced sequence, recording, and self-supervised relative-positioning samplers.</span>
            </a>
            <a class="mod-row" href="api.html#data-utils">
              <span class="mod-name bd-mono">braindecode.datautil</span>
              <span class="mod-desc">Dataset serialization and automatic signal-property inference.</span>
            </a>
          </div>

          <div class="mod-col">
            <h3 class="mod-col-title">Models, Layers &amp; Kernels</h3>
            <a class="mod-row" href="api.html#models">
              <span class="mod-name bd-mono">braindecode.models</span>
              <span class="mod-desc">103 peer-reviewed backbones and 48 foundation encoders on EEGModuleMixin.</span>
            </a>
            <a class="mod-row" href="api.html#modules">
              <span class="mod-name bd-mono">braindecode.modules</span>
              <span class="mod-desc">ChannelLayer montage adapters, RVQ codebooks, SEANet, and constrained convs.</span>
            </a>
            <a class="mod-row" href="api.html#functional">
              <span class="mod-name bd-mono">braindecode.functional</span>
              <span class="mod-desc">Exact bipolar derivations, masked_group_norm, FFT conv1d, and rotary angles.</span>
            </a>
            <a class="mod-row" href="api.html#augmentation">
              <span class="mod-name bd-mono">braindecode.augmentation</span>
              <span class="mod-desc">20+ differentiable time, frequency-surrogate, and 3D sensor-rotation transforms.</span>
            </a>
          </div>

          <div class="mod-col">
            <h3 class="mod-col-title">Train, Score &amp; Interpret</h3>
            <a class="mod-row" href="api.html#classifier">
              <span class="mod-name bd-mono">EEGClassifier / EEGRegressor</span>
              <span class="mod-desc">skorch estimators bringing scikit-learn fit(), predict(), and CV to PyTorch.</span>
            </a>
            <a class="mod-row" href="api.html#training">
              <span class="mod-name bd-mono">braindecode.training</span>
              <span class="mod-desc">Cropped decoding, chunked trial inference, f1_event, and szcore_event_scores.</span>
            </a>
            <a class="mod-row" href="api.html#visualization">
              <span class="mod-name bd-mono">braindecode.visualization</span>
              <span class="mod-desc">Time and amplitude-spectrum attributions, randomization checks, and topomaps.</span>
            </a>
            <a class="mod-row" href="api.html#utils">
              <span class="mod-name bd-mono">braindecode.util</span>
              <span class="mod-desc">Deterministic random-seed setup and MNE standard montage resolution.</span>
            </a>
          </div>
        </div>
      </div>
    </section>

    <section class="section section-alt tut-section">
      <div class="container">
        <div class="sec-head">
          <h2 class="section-title">Runnable tutorials built from executed notebooks.</h2>
          <a class="sec-action" href="auto_examples/index.html">Browse all tutorials <span aria-hidden="true">→</span></a>
        </div>

        <div class="tut-grid">
          <a class="tut-card" href="auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html">
            <img class="tut-thumb" src="_images/sphx_glr_plot_bcic_iv_2a_moabb_trial_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Motor Imagery · EEG</span>
            <h3 class="tut-title">Trialwise decoding on BCI Competition IV-2a</h3>
          </a>
          <a class="tut-card" href="auto_examples/model_building/plot_load_pretrained_models.html">
            <img class="tut-thumb" src="_images/sphx_glr_plot_load_pretrained_models_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Foundation Hub</span>
            <h3 class="tut-title">Load, adapt, and audit pretrained foundation models</h3>
          </a>
          <a class="tut-card" href="auto_examples/model_building/plot_heterogeneous_montage_training.html">
            <img class="tut-thumb" src="_images/sphx_glr_plot_heterogeneous_montage_training_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Channel Geometry</span>
            <h3 class="tut-title">Train position-aware models across heterogeneous montages</h3>
          </a>
          <a class="tut-card" href="auto_examples/applied_examples/plot_seizure_alarm_chbmit.html">
            <img class="tut-thumb" src="_images/sphx_glr_plot_seizure_alarm_chbmit_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Clinical Epilepsy</span>
            <h3 class="tut-title">CHB-MIT seizure alarm evaluated with SzCORE</h3>
          </a>
          <a class="tut-card" href="auto_examples/applied_examples/plot_sleep_staging_chambon2018.html">
            <img class="tut-thumb" src="_images/sphx_glr_plot_sleep_staging_chambon2018_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Sleep PSG</span>
            <h3 class="tut-title">Clinical polysomnography staging on Sleep Physionet</h3>
          </a>
          <a class="tut-card" href="auto_examples/advanced_training/plot_interpretability.html">
            <img class="tut-thumb" src="_images/sphx_glr_plot_interpretability_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Attribution &amp; XAI</span>
            <h3 class="tut-title">Amplitude-spectrum gradients and sanity-check protocols</h3>
          </a>
        </div>
      </div>
    </section>

    <section class="section prov-section" id="ecosystem">
      <div class="container">
        <div class="sec-head">
          <h2 class="section-title">Grounded in peer-reviewed neurotechnology and open standards.</h2>
          <p class="section-sub">Cite canonical references in papers and connect directly with the Python neuroscience stack.</p>
        </div>

        <div class="prov-grid">
          <div class="prov-left">
            <article class="cite-card bd-pane">
              <div class="cite-meta bd-mono">HUMAN BRAIN MAPPING · 2017 · DOI 10.1002/hbm.23730</div>
              <h3 class="cite-title">Deep learning with convolutional neural networks for EEG decoding and visualization</h3>
              <p class="cite-authors">Robin Tibor Schirrmeister, Jost Tobias Springenberg, Lukas D. J. Fiederer, Martin Glasstetter, Katharina Eggensperger, Michael Tangermann, Frank Hutter, Wolfram Burgard, Tonio Ball · Software DOI: <a href="https://doi.org/10.5281/zenodo.8214376" target="_blank" rel="noopener">10.5281/zenodo.8214376</a></p>
              <div class="cite-actions">
                <span class="copy-wrap">
                  <button type="button" class="btn btn-sm" data-copy="@article{HBM:HBM23730,&#10;  author = {Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio},&#10;  title = {Deep learning with convolutional neural networks for EEG decoding and visualization},&#10;  journal = {Human Brain Mapping},&#10;  year = {2017},&#10;  doi = {10.1002/hbm.23730}&#10;}" aria-label="Copy BibTeX citation for Schirrmeister et al. 2017">Copy BibTeX</button>
                  <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
                </span>
                <a class="sec-action" href="cite.html">All software &amp; paper citations <span aria-hidden="true">→</span></a>
              </div>
            </article>

            <article class="agent-card bd-pane">
              <div class="cite-meta bd-mono">MACHINE-READABLE DOCUMENTATION</div>
              <h3 class="cite-title">Commit-pinned Markdown &amp; llms.txt for coding agents</h3>
              <p class="cite-authors">Every documentation build publishes version-scoped Markdown mirrors of installation, API contracts, model specifications, and tutorials so AI coding tools read exact signatures.</p>
              <div class="cite-actions">
                <a class="btn btn-sm bd-mono" href="llms.txt">llms.txt</a>
                <a class="sec-action" href="agent_docs.html">Agent build specification <span aria-hidden="true">→</span></a>
              </div>
            </article>
          </div>

          <div class="eco-ledger">
            <a class="eco-cell" href="https://mne.tools/" target="_blank" rel="noopener">
              <span class="eco-role bd-mono">Signal foundation</span>
              <h3 class="eco-name">MNE-Python</h3>
              <p class="eco-desc">Native interop with mne.io.Raw, mne.Epochs, and 3D sensor montages.</p>
            </a>
            <a class="eco-cell" href="https://moabb.neurotechx.com/" target="_blank" rel="noopener">
              <span class="eco-role bd-mono">Benchmark suite</span>
              <h3 class="eco-name">MOABB</h3>
              <p class="eco-desc">Reproducible BCI evaluation wrapping 150+ open datasets out of the box.</p>
            </a>
            <a class="eco-cell" href="https://eegdash.org/" target="_blank" rel="noopener">
              <span class="eco-role bd-mono">Data archive</span>
              <h3 class="eco-name">EEGDash</h3>
              <p class="eco-desc">700+ BIDS-curated EEG, MEG, iEEG, fNIRS, and EMG corpora with Zarr streaming.</p>
            </a>
            <a class="eco-cell" href="https://skorch.readthedocs.io/" target="_blank" rel="noopener">
              <span class="eco-role bd-mono">Estimator API</span>
              <h3 class="eco-name">skorch</h3>
              <p class="eco-desc">scikit-learn compatible fit(), predict(), and GridSearchCV on PyTorch modules.</p>
            </a>
            <a class="eco-cell" href="https://facebookresearch.github.io/neuroai/" target="_blank" rel="noopener">
              <span class="eco-role bd-mono">Downstream suite</span>
              <h3 class="eco-name">Meta Neuro AI</h3>
              <p class="eco-desc">FAIR neuro-AI ecosystem whose NeuralTrain library builds on Braindecode backbones.</p>
            </a>
            <a class="eco-cell" href="https://github.com/aeon-toolkit/aeon-neuro" target="_blank" rel="noopener">
              <span class="eco-role bd-mono">Downstream toolkit</span>
              <h3 class="eco-name">aeon-neuro</h3>
              <p class="eco-desc">Time-series classification extension wrapping Braindecode neural decoders.</p>
            </a>
          </div>
        </div>
      </div>
    </section>

    <footer class="site-footer">
      <div class="container">
        <div class="site-footer-cols">
          <div class="footer-brand-col">
            <img class="footer-logo" src="_static/braindecode_long.svg" alt="Braindecode" width="171" height="28" loading="lazy" decoding="async"/>
            <p class="site-footer-blurb">Open-source PyTorch library for decoding raw EEG, iEEG, MEG, fNIRS, and sEMG signals. Built on PyTorch, MNE-Python, and skorch.</p>
          </div>
          <div>
            <h3>Get started</h3>
            <ul>
              <li><a href="install/install.html">Installation</a></li>
              <li><a href="auto_examples/index.html">Tutorials</a></li>
              <li><a href="user_guide/channel_strategies.html">Channel strategies</a></li>
              <li><a href="user_guide/cropped_inference.html">Cropped inference</a></li>
              <li><a href="api.html">API reference</a></li>
            </ul>
          </div>
          <div>
            <h3>Models &amp; Hub</h3>
            <ul>
              <li><a href="models/models.html">Model overview</a></li>
              <li><a href="models/models_table.html">Models summary table</a></li>
              <li><a href="models/categorization/lbm.html">Foundation models</a></li>
              <li><a href="https://huggingface.co/braindecode" target="_blank" rel="noopener">Hugging Face Hub</a></li>
            </ul>
          </div>
          <div>
            <h3>Community</h3>
            <ul>
              <li><a href="https://github.com/braindecode/braindecode" target="_blank" rel="noopener">GitHub repository</a></li>
              <li><a href="https://github.com/braindecode/braindecode/discussions" target="_blank" rel="noopener">Discussions</a></li>
              <li><a href="whats_new.html">Release notes</a></li>
              <li><a href="cite.html">Cite Braindecode</a></li>
              <li><a href="agent_docs.html">Agent docs &amp; llms.txt</a></li>
            </ul>
          </div>
          <div>
            <h3>Ecosystem</h3>
            <ul>
              <li><a href="https://mne.tools/" target="_blank" rel="noopener">MNE-Python</a></li>
              <li><a href="https://moabb.neurotechx.com/" target="_blank" rel="noopener">MOABB</a></li>
              <li><a href="https://eegdash.org/" target="_blank" rel="noopener">EEGDash</a></li>
              <li><a href="https://pytorch.org/" target="_blank" rel="noopener">PyTorch</a></li>
              <li><a href="help.html">Help &amp; support</a></li>
            </ul>
          </div>
        </div>
        <div class="site-footer-bottom">
          <span>© 2017–2026 The Braindecode Developers · BSD-3-Clause License</span>
          <span>Maintained across UC San Diego, Inria, University of Freiburg, Donders Institute, and Yneuro.</span>
        </div>
      </div>
    </footer>

    </div>

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
