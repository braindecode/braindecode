:html_theme.sidebar_secondary.remove: true

.. meta::
    :description: Open-source PyTorch library for decoding raw EEG, iEEG, ECoG, MEG, fNIRS, and sEMG brain signals with 103 published architectures, 48 pretrained foundation models, and 1,100+ open BIDS and BCI datasets.
    :keywords: EEG deep learning, Braindecode, PyTorch EEG, brain foundation models, iEEG, ECoG, MEG, fNIRS, sEMG, BCI, MNE-Python, MOABB, EEGDash, SPD Learn, EEGPrep, Hugging Face Hub

.. raw:: html

    <div class="bd-landing">

###############################################################
 Braindecode: PyTorch Deep Learning for EEG, iEEG, MEG & fNIRS
###############################################################

.. raw:: html

    <section class="hero">
      <div class="container">
        <div class="hero-center">
          <div class="hero-eyebrow">
            <img class="hero-eyebrow-mark only-light" src="_static/braindecode_small.svg" alt="" width="34" height="18"/>
            <img class="hero-eyebrow-mark only-dark" src="_static/braindecode_small_dark.svg" alt="" width="34" height="18"/>
            <span class="hero-eyebrow-brand">BRAINDECODE</span>
            <span class="hero-eyebrow-sep" aria-hidden="true"></span>
            <span>OPEN-SOURCE PYTORCH FOR BRAIN SIGNALS · SINCE 2017</span>
          </div>

          <h2 class="hero-title">Decode Brain Signals<br/>with <span class="accent">Braindecode</span></h2>

          <p class="hero-lede">
            A PyTorch-native toolbox for end-to-end neural decoding. Load a dataset, pick a published model, train. Same scikit-learn API you already know.
          </p>

          <div class="hero-cta-row">
            <a class="btn btn-primary" href="install/install.html">Get started <span aria-hidden="true">→</span></a>
            <a class="btn" href="auto_examples/index.html">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><polygon points="6 4 20 12 6 20 6 4"/></svg>
              <span>Tutorials</span>
            </a>
            <a class="btn" href="#model-zoo">Explore 103 models</a>
            <a class="btn" href="https://github.com/braindecode/braindecode" target="_blank" rel="noopener">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 0 0 7.86 10.92c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.38-3.88-1.38-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.2 1.77 1.2 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11.1 11.1 0 0 1 5.8 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.12 3.05.74.81 1.18 1.83 1.18 3.09 0 4.41-2.69 5.38-5.26 5.67.42.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"/></svg>
              <span>GitHub</span>
            </a>
          </div>

          <div class="hero-install-row">
            <span class="copy-wrap">
              <button type="button" class="hero-install bd-groove bd-mono" data-copy="pip install braindecode" aria-label="Copy pip install braindecode">
                <span class="prompt" aria-hidden="true">$</span>
                <code>pip install braindecode</code>
                <span class="copy-icon" aria-hidden="true">
                  <svg class="icon-clipboard" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                  <svg class="icon-check" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
                </span>
              </button>
              <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
            </span>
          </div>

          <p class="hero-audience">
            For <strong>neuroscientists and clinicians</strong> who want to work with deep learning, and
            <strong>machine learning researchers</strong> who want to work with neurophysiological data.
          </p>

          <div class="proof-bar" aria-label="Project adoption metrics">
            <a class="proof-item" href="https://pepy.tech/projects/braindecode" target="_blank" rel="noopener">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              <span><b id="bd-downloads-num">1.4M+</b> PYPI DOWNLOADS</span>
            </a>
            <span class="proof-div" aria-hidden="true"></span>
            <a class="proof-item" href="https://huggingface.co/braindecode" target="_blank" rel="noopener">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/></svg>
              <span><b>402K/MO</b> HUGGING FACE · <b>141</b> REPOS</span>
            </a>
            <span class="proof-div" aria-hidden="true"></span>
            <a class="proof-item" href="https://github.com/braindecode/braindecode" target="_blank" rel="noopener">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 0 0 7.86 10.92c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.38-3.88-1.38-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.2 1.77 1.2 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11.1 11.1 0 0 1 5.8 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.12 3.05.74.81 1.18 1.83 1.18 3.09 0 4.41-2.69 5.38-5.26 5.67.42.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"/></svg>
              <span><b>1.3K+</b> GITHUB STARS</span>
            </a>
            <span class="proof-div" aria-hidden="true"></span>
            <a class="proof-item" href="cite.html">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
              <span><b>4,600+</b> CITATIONS · HBM 2017</span>
            </a>
          </div>
        </div>

        <div class="hero-art-stage">
          <svg class="hero-art-svg" viewBox="0 0 1120 338" role="img" aria-label="Animated Braindecode pipeline: continuous raw brainwaves flow out of the cortex through a sliding epoch window into a deep neural network and decoded output">
            <defs>
              <clipPath id="bd-wave-clip">
                <rect x="175" y="42" width="568" height="264"/>
              </clipPath>
              <linearGradient id="bd-wave-fade" x1="175" y1="0" x2="743" y2="0" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stop-color="#ffffff" stop-opacity="0.15"/>
                <stop offset="5%" stop-color="#ffffff" stop-opacity="1"/>
                <stop offset="95%" stop-color="#ffffff" stop-opacity="1"/>
                <stop offset="100%" stop-color="#ffffff" stop-opacity="0.25"/>
              </linearGradient>
              <mask id="bd-wave-mask">
                <rect x="175" y="42" width="568" height="264" fill="url(#bd-wave-fade)"/>
              </mask>
              <radialGradient id="bd-readout-aura" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#e5964f" stop-opacity="0.48"/>
                <stop offset="60%" stop-color="#b46a2c" stop-opacity="0.16"/>
                <stop offset="100%" stop-color="#b46a2c" stop-opacity="0"/>
              </radialGradient>
            </defs>

            <g class="hero-guide-lines" aria-hidden="true">
              <line x1="180" y1="74" x2="742" y2="74"/>
              <line x1="180" y1="124" x2="742" y2="124"/>
              <line x1="180" y1="174" x2="742" y2="174"/>
              <line x1="180" y1="224" x2="742" y2="224"/>
              <line x1="180" y1="274" x2="742" y2="274"/>
            </g>

            <g clip-path="url(#bd-wave-clip)" mask="url(#bd-wave-mask)">
              <g class="hero-wave-runner">
                <path class="hero-wave ch1" d="M 192 78.82 L 196 83.46 L 200 83.11 L 204 78.48 L 208 72.46 L 212 68.34 L 216 68.07 L 220 71.35 L 224 76.15 L 228 80.02 L 232 81.4 L 236 80.24 L 240 77.63 L 244 74.94 L 248 73.08 L 252 72.29 L 256 72.42 L 260 73.31 L 264 74.88 L 268 76.9 L 272 78.7 L 276 79.22 L 280 77.6 L 284 73.9 L 288 69.44 L 292 66.41 L 296 66.67 L 300 70.5 L 304 76.17 L 308 80.61 L 312 81.05 L 316 76.76 L 320 69.66 L 324 63.46 L 328 61.63 L 332 65.4 L 336 72.94 L 340 80.23 L 344 83.33 L 348 80.6 L 352 73.58 L 356 66.23 L 360 62.61 L 364 64.68 L 368 71.27 L 372 78.82 L 376 83.46 L 380 83.11 L 384 78.48 L 388 72.46 L 392 68.34 L 396 68.07 L 400 71.35 L 404 76.15 L 408 80.02 L 412 81.4 L 416 80.24 L 420 77.63 L 424 74.94 L 428 73.08 L 432 72.29 L 436 72.42 L 440 73.31 L 444 74.88 L 448 76.9 L 452 78.7 L 456 79.22 L 460 77.6 L 464 73.9 L 468 69.44 L 472 66.41 L 476 66.67 L 480 70.5 L 484 76.17 L 488 80.61 L 492 81.05 L 496 76.76 L 500 69.66 L 504 63.46 L 508 61.63 L 512 65.4 L 516 72.94 L 520 80.23 L 524 83.33 L 528 80.6 L 532 73.58 L 536 66.23 L 540 62.61 L 544 64.68 L 548 71.27 L 552 78.82 L 556 83.46 L 560 83.11 L 564 78.48 L 568 72.46 L 572 68.34 L 576 68.07 L 580 71.35 L 584 76.15 L 588 80.02 L 592 81.4 L 596 80.24 L 600 77.63 L 604 74.94 L 608 73.08 L 612 72.29 L 616 72.42 L 620 73.31 L 624 74.88 L 628 76.9 L 632 78.7 L 636 79.22 L 640 77.6 L 644 73.9 L 648 69.44 L 652 66.41 L 656 66.67 L 660 70.5 L 664 76.17 L 668 80.61 L 672 81.05 L 676 76.76 L 680 69.66 L 684 63.46 L 688 61.63 L 692 65.4 L 696 72.94 L 700 80.23 L 704 83.33 L 708 80.6 L 712 73.58 L 716 66.23 L 720 62.61 L 724 64.68 L 728 71.27 L 732 78.82 L 736 83.46 L 740 83.11 L 744 78.48 L 748 72.46 L 752 68.34 L 756 68.07 L 760 71.35 L 764 76.15 L 768 80.02 L 772 81.4 L 776 80.24 L 780 77.63 L 784 74.94 L 788 73.08 L 792 72.29 L 796 72.42 L 800 73.31 L 804 74.88 L 808 76.9 L 812 78.7 L 816 79.22 L 820 77.6 L 824 73.9 L 828 69.44 L 832 66.41 L 836 66.67 L 840 70.5 L 844 76.17 L 848 80.61 L 852 81.05 L 856 76.76 L 860 69.66 L 864 63.46 L 868 61.63 L 872 65.4 L 876 72.94 L 880 80.23 L 884 83.33 L 888 80.6 L 892 73.58 L 896 66.23 L 900 62.61 L 904 64.68 L 908 71.27 L 912 78.82"/>
                <path class="hero-wave ch2" d="M 192 134.04 L 196 133.2 L 200 130.32 L 204 128.35 L 208 128.95 L 212 131.1 L 216 132.05 L 220 129.9 L 224 125.28 L 228 120.93 L 232 119.32 L 236 120.5 L 240 122.14 L 244 121.6 L 248 118.32 L 252 114.39 L 256 112.77 L 260 114.76 L 264 118.92 L 268 122.38 L 272 123.34 L 276 122.65 L 280 122.93 L 284 126.1 L 288 131.5 L 292 136.28 L 296 137.67 L 300 135.28 L 304 131.31 L 308 128.56 L 312 128.03 L 316 128.13 L 320 126.29 L 324 121.48 L 328 115.37 L 332 111.12 L 336 110.82 L 340 113.8 L 344 117.28 L 348 118.82 L 352 118.39 L 356 118.2 L 360 120.58 L 364 125.69 L 368 131.17 L 372 134.04 L 376 133.2 L 380 130.32 L 384 128.35 L 388 128.95 L 392 131.1 L 396 132.05 L 400 129.9 L 404 125.28 L 408 120.93 L 412 119.32 L 416 120.5 L 420 122.14 L 424 121.6 L 428 118.32 L 432 114.39 L 436 112.77 L 440 114.76 L 444 118.92 L 448 122.38 L 452 123.34 L 456 122.65 L 460 122.93 L 464 126.1 L 468 131.5 L 472 136.28 L 476 137.67 L 480 135.28 L 484 131.31 L 488 128.56 L 492 128.03 L 496 128.13 L 500 126.29 L 504 121.48 L 508 115.37 L 512 111.12 L 516 110.82 L 520 113.8 L 524 117.28 L 528 118.82 L 532 118.39 L 536 118.2 L 540 120.58 L 544 125.69 L 548 131.17 L 552 134.04 L 556 133.2 L 560 130.32 L 564 128.35 L 568 128.95 L 572 131.1 L 576 132.05 L 580 129.9 L 584 125.28 L 588 120.93 L 592 119.32 L 596 120.5 L 600 122.14 L 604 121.6 L 608 118.32 L 612 114.39 L 616 112.77 L 620 114.76 L 624 118.92 L 628 122.38 L 632 123.34 L 636 122.65 L 640 122.93 L 644 126.1 L 648 131.5 L 652 136.28 L 656 137.67 L 660 135.28 L 664 131.31 L 668 128.56 L 672 128.03 L 676 128.13 L 680 126.29 L 684 121.48 L 688 115.37 L 692 111.12 L 696 110.82 L 700 113.8 L 704 117.28 L 708 118.82 L 712 118.39 L 716 118.2 L 720 120.58 L 724 125.69 L 728 131.17 L 732 134.04 L 736 133.2 L 740 130.32 L 744 128.35 L 748 128.95 L 752 131.1 L 756 132.05 L 760 129.9 L 764 125.28 L 768 120.93 L 772 119.32 L 776 120.5 L 780 122.14 L 784 121.6 L 788 118.32 L 792 114.39 L 796 112.77 L 800 114.76 L 804 118.92 L 808 122.38 L 812 123.34 L 816 122.65 L 820 122.93 L 824 126.1 L 828 131.5 L 832 136.28 L 836 137.67 L 840 135.28 L 844 131.31 L 848 128.56 L 852 128.03 L 856 128.13 L 860 126.29 L 864 121.48 L 868 115.37 L 872 111.12 L 876 110.82 L 880 113.8 L 884 117.28 L 888 118.82 L 892 118.39 L 896 118.2 L 900 120.58 L 904 125.69 L 908 131.17 L 912 134.04"/>
                <path class="hero-wave ch3" d="M 192 180.94 L 196 183.26 L 200 177.73 L 204 169.12 L 208 165.87 L 212 171.17 L 216 179.5 L 220 182.51 L 224 177.75 L 228 170.8 L 232 168.68 L 236 172.15 L 240 175.41 L 244 173.39 L 248 167.99 L 252 166.07 L 256 171.4 L 260 179.72 L 264 182.85 L 268 177.49 L 272 169.0 L 276 165.91 L 280 171.01 L 284 178.7 L 288 181.4 L 292 177.66 L 296 173.12 L 300 173.49 L 304 177.82 L 308 179.68 L 312 174.99 L 316 167.34 L 320 164.55 L 324 170.03 L 328 178.64 L 332 181.83 L 336 176.59 L 340 168.48 L 344 165.55 L 348 169.8 L 352 175.66 L 356 176.74 L 360 173.1 L 364 170.79 L 368 174.33 L 372 180.94 L 376 183.26 L 380 177.73 L 384 169.12 L 388 165.87 L 392 171.17 L 396 179.5 L 400 182.51 L 404 177.75 L 408 170.8 L 412 168.68 L 416 172.15 L 420 175.41 L 424 173.39 L 428 167.99 L 432 166.07 L 436 171.4 L 440 179.72 L 444 182.85 L 448 177.49 L 452 169.0 L 456 165.91 L 460 171.01 L 464 178.7 L 468 181.4 L 472 177.66 L 476 173.12 L 480 173.49 L 484 177.82 L 488 179.68 L 492 174.99 L 496 167.34 L 500 164.55 L 504 170.03 L 508 178.64 L 512 181.83 L 516 176.59 L 520 168.48 L 524 165.55 L 528 169.8 L 532 175.66 L 536 176.74 L 540 173.1 L 544 170.79 L 548 174.33 L 552 180.94 L 556 183.26 L 560 177.73 L 564 169.12 L 568 165.87 L 572 171.17 L 576 179.5 L 580 182.51 L 584 177.75 L 588 170.8 L 592 168.68 L 596 172.15 L 600 175.41 L 604 173.39 L 608 167.99 L 612 166.07 L 616 171.4 L 620 179.72 L 624 182.85 L 628 177.49 L 632 169.0 L 636 165.91 L 640 171.01 L 644 178.7 L 648 181.4 L 652 177.66 L 656 173.12 L 660 173.49 L 664 177.82 L 668 179.68 L 672 174.99 L 676 167.34 L 680 164.55 L 684 170.03 L 688 178.64 L 692 181.83 L 696 176.59 L 700 168.48 L 704 165.55 L 708 169.8 L 712 175.66 L 716 176.74 L 720 173.1 L 724 170.79 L 728 174.33 L 732 180.94 L 736 183.26 L 740 177.73 L 744 169.12 L 748 165.87 L 752 171.17 L 756 179.5 L 760 182.51 L 764 177.75 L 768 170.8 L 772 168.68 L 776 172.15 L 780 175.41 L 784 173.39 L 788 167.99 L 792 166.07 L 796 171.4 L 800 179.72 L 804 182.85 L 808 177.49 L 812 169.0 L 816 165.91 L 820 171.01 L 824 178.7 L 828 181.4 L 832 177.66 L 836 173.12 L 840 173.49 L 844 177.82 L 848 179.68 L 852 174.99 L 856 167.34 L 860 164.55 L 864 170.03 L 868 178.64 L 872 181.83 L 876 176.59 L 880 168.48 L 884 165.55 L 888 169.8 L 892 175.66 L 896 176.74 L 900 173.1 L 904 170.79 L 908 174.33 L 912 180.94"/>
                <path class="hero-wave ch4" d="M 192 232.81 L 196 227.36 L 200 216.16 L 204 221.06 L 208 235.36 L 212 233.31 L 216 216.92 L 220 213.74 L 224 227.83 L 228 233.02 L 232 220.34 L 236 212.54 L 240 221.83 L 244 230.21 L 248 225.01 L 252 219.79 L 256 224.98 L 260 229.01 L 264 224.34 L 268 222.74 L 272 229.15 L 276 229.25 L 280 218.77 L 284 215.84 L 288 227.16 L 292 231.92 L 296 219.21 L 300 210.72 L 304 222.77 L 308 235.6 L 312 227.97 L 316 214.96 L 320 220.13 L 324 233.74 L 328 232.28 L 332 220.42 L 336 219.11 L 340 226.79 L 344 226.55 L 348 219.88 L 352 220.7 L 356 225.71 L 360 222.39 L 364 217.09 L 368 223.54 L 372 232.81 L 376 227.36 L 380 216.16 L 384 221.06 L 388 235.36 L 392 233.31 L 396 216.92 L 400 213.74 L 404 227.83 L 408 233.02 L 412 220.34 L 416 212.54 L 420 221.83 L 424 230.21 L 428 225.01 L 432 219.79 L 436 224.98 L 440 229.01 L 444 224.34 L 448 222.74 L 452 229.15 L 456 229.25 L 460 218.77 L 464 215.84 L 468 227.16 L 472 231.92 L 476 219.21 L 480 210.72 L 484 222.77 L 488 235.6 L 492 227.97 L 496 214.96 L 500 220.13 L 504 233.74 L 508 232.28 L 512 220.42 L 516 219.11 L 520 226.79 L 524 226.55 L 528 219.88 L 532 220.7 L 536 225.71 L 540 222.39 L 544 217.09 L 548 223.54 L 552 232.81 L 556 227.36 L 560 216.16 L 564 221.06 L 568 235.36 L 572 233.31 L 576 216.92 L 580 213.74 L 584 227.83 L 588 233.02 L 592 220.34 L 596 212.54 L 600 221.83 L 604 230.21 L 608 225.01 L 612 219.79 L 616 224.98 L 620 229.01 L 624 224.34 L 628 222.74 L 632 229.15 L 636 229.25 L 640 218.77 L 644 215.84 L 648 227.16 L 652 231.92 L 656 219.21 L 660 210.72 L 664 222.77 L 668 235.6 L 672 227.97 L 676 214.96 L 680 220.13 L 684 233.74 L 688 232.28 L 692 220.42 L 696 219.11 L 700 226.79 L 704 226.55 L 708 219.88 L 712 220.7 L 716 225.71 L 720 222.39 L 724 217.09 L 728 223.54 L 732 232.81 L 736 227.36 L 740 216.16 L 744 221.06 L 748 235.36 L 752 233.31 L 756 216.92 L 760 213.74 L 764 227.83 L 768 233.02 L 772 220.34 L 776 212.54 L 780 221.83 L 784 230.21 L 788 225.01 L 792 219.79 L 796 224.98 L 800 229.01 L 804 224.34 L 808 222.74 L 812 229.15 L 816 229.25 L 820 218.77 L 824 215.84 L 828 227.16 L 832 231.92 L 836 219.21 L 840 210.72 L 844 222.77 L 848 235.6 L 852 227.97 L 856 214.96 L 860 220.13 L 864 233.74 L 868 232.28 L 872 220.42 L 876 219.11 L 880 226.79 L 884 226.55 L 888 219.88 L 892 220.7 L 896 225.71 L 900 222.39 L 904 217.09 L 908 223.54 L 912 232.81"/>
                <path class="hero-wave ch5" d="M 192 284.47 L 196 286.37 L 200 280.81 L 204 275.21 L 208 275.55 L 212 276.97 L 216 272.51 L 220 265.33 L 224 264.14 L 228 269.3 L 232 272.38 L 236 270.06 L 240 269.18 L 244 274.87 L 248 281.37 L 252 280.84 L 256 275.77 L 260 274.54 L 264 277.52 L 268 276.72 L 272 269.5 L 276 263.62 L 280 265.12 L 284 269.01 L 288 267.94 L 292 264.58 L 296 267.05 L 300 275.13 L 304 280.02 L 308 278.13 L 312 276.35 L 316 280.04 L 320 283.87 L 324 280.54 L 328 273.11 L 332 270.42 L 336 273.17 L 340 273.46 L 344 268.49 L 348 265.65 L 352 270.48 L 356 277.41 L 360 278.58 L 364 276.21 L 368 278.23 L 372 284.47 L 376 286.37 L 380 280.81 L 384 275.21 L 388 275.55 L 392 276.97 L 396 272.51 L 400 265.33 L 404 264.14 L 408 269.3 L 412 272.38 L 416 270.06 L 420 269.18 L 424 274.87 L 428 281.37 L 432 280.84 L 436 275.77 L 440 274.54 L 444 277.52 L 448 276.72 L 452 269.5 L 456 263.62 L 460 265.12 L 464 269.01 L 468 267.94 L 472 264.58 L 476 267.05 L 480 275.13 L 484 280.02 L 488 278.13 L 492 276.35 L 496 280.04 L 500 283.87 L 504 280.54 L 508 273.11 L 512 270.42 L 516 273.17 L 520 273.46 L 524 268.49 L 528 265.65 L 532 270.48 L 536 277.41 L 540 278.58 L 544 276.21 L 548 278.23 L 552 284.47 L 556 286.37 L 560 280.81 L 564 275.21 L 568 275.55 L 572 276.97 L 576 272.51 L 580 265.33 L 584 264.14 L 588 269.3 L 592 272.38 L 596 270.06 L 600 269.18 L 604 274.87 L 608 281.37 L 612 280.84 L 616 275.77 L 620 274.54 L 624 277.52 L 628 276.72 L 632 269.5 L 636 263.62 L 640 265.12 L 644 269.01 L 648 267.94 L 652 264.58 L 656 267.05 L 660 275.13 L 664 280.02 L 668 278.13 L 672 276.35 L 676 280.04 L 680 283.87 L 684 280.54 L 688 273.11 L 692 270.42 L 696 273.17 L 700 273.46 L 704 268.49 L 708 265.65 L 712 270.48 L 716 277.41 L 720 278.58 L 724 276.21 L 728 278.23 L 732 284.47 L 736 286.37 L 740 280.81 L 744 275.21 L 748 275.55 L 752 276.97 L 756 272.51 L 760 265.33 L 764 264.14 L 768 269.3 L 772 272.38 L 776 270.06 L 780 269.18 L 784 274.87 L 788 281.37 L 792 280.84 L 796 275.77 L 800 274.54 L 804 277.52 L 808 276.72 L 812 269.5 L 816 263.62 L 820 265.12 L 824 269.01 L 828 267.94 L 832 264.58 L 836 267.05 L 840 275.13 L 844 280.02 L 848 278.13 L 852 276.35 L 856 280.04 L 860 283.87 L 864 280.54 L 868 273.11 L 872 270.42 L 876 273.17 L 880 273.46 L 884 268.49 L 888 265.65 L 892 270.48 L 896 277.41 L 900 278.58 L 904 276.21 L 908 278.23 L 912 284.47"/>
              </g>
            </g>

            <g class="hero-ch-tags bd-mono" aria-hidden="true">
              <text x="194" y="61">EEG</text>
              <text x="194" y="111">PSG</text>
              <text x="194" y="161">MEG</text>
              <text x="194" y="211">iEEG</text>
              <text x="194" y="261">sEMG · fNIRS</text>
            </g>

            <g class="bd-hero-window" aria-hidden="true">
              <rect x="386" y="48" width="132" height="248" rx="8" class="bd-win-box"/>
              <line x1="386" y1="42" x2="386" y2="54" class="bd-win-tick"/>
              <line x1="518" y1="42" x2="518" y2="54" class="bd-win-tick"/>
              <text x="452" y="36" text-anchor="middle" class="bd-win-lbl bd-mono">epoch (C × T)</text>
            </g>

            <g class="hero-cortex" transform="translate(30, 55) scale(2.38)">
              <path class="hero-cortex-fill" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" d="M 60.93 13.09 C 60.19 12.72, 59.87 11.80, 59.40 10.99 C 58.92 10.19, 58.30 9.51, 57.60 8.90 C 56.22 7.70, 54.57 6.81, 52.81 6.26 C 49.48 5.23, 45.78 5.45, 42.60 6.84 C 40.92 7.57, 39.39 8.63, 38.19 10.07 C 37.58 10.79, 37.06 11.60, 36.32 12.06 C 35.58 12.53, 34.62 12.64, 33.68 12.68 C 32.74 12.73, 31.82 12.71, 30.94 12.87 C 30.05 13.03, 29.19 13.38, 28.36 13.79 C 26.87 14.54, 25.45 15.52, 24.74 17.05 C 24.35 17.90, 24.18 18.92, 23.64 19.57 C 23.32 19.97, 22.86 20.23, 22.33 20.35 C 21.97 20.42, 21.57 20.43, 21.20 20.46 C 20.26 20.54, 19.45 20.76, 18.61 21.18 C 17.78 21.60, 16.92 22.23, 16.44 22.99 C 15.49 24.50, 16.06 26.53, 15.77 28.36 C 15.63 29.28, 15.28 30.15, 14.61 30.73 C 14.08 31.20, 13.34 31.48, 12.64 31.78 C 11.77 32.15, 10.95 32.56, 10.29 33.16 C 9.63 33.75, 9.13 34.54, 8.70 35.38 C 8.28 36.23, 7.93 37.14, 7.84 38.01 C 7.77 38.71, 7.87 39.38, 8.14 40.09 C 8.47 40.98, 9.07 41.92, 9.06 42.76 C 9.06 43.60, 8.45 44.33, 7.86 45.09 C 6.80 46.45, 5.81 47.86, 5.64 49.45 C 5.45 51.20, 6.26 53.17, 7.64 54.41 C 8.33 55.04, 9.16 55.48, 9.67 56.21 C 10.19 56.93, 10.39 57.94, 10.33 58.85 C 10.26 59.77, 9.91 60.59, 9.84 61.51 C 9.76 62.43, 9.96 63.43, 10.35 64.27 C 11.05 65.77, 12.41 66.70, 13.72 67.77 C 14.45 68.37, 15.17 69.00, 15.45 69.82 C 15.74 70.63, 15.58 71.63, 15.54 72.59 C 15.46 74.52, 15.81 76.33, 17.09 77.65 C 17.73 78.30, 18.60 78.84, 19.49 79.13 C 20.37 79.42, 21.27 79.46, 22.16 79.65 C 22.70 79.77, 23.22 79.93, 23.56 80.30 C 23.79 80.55, 23.93 80.89, 24.07 81.24 C 24.34 81.94, 24.59 82.65, 24.96 83.28 C 25.90 84.86, 27.59 85.86, 29.30 86.53 C 30.15 86.87, 31.01 87.12, 31.93 87.18 C 32.85 87.25, 33.82 87.13, 34.72 87.29 C 35.61 87.45, 36.42 87.90, 37.09 88.53 C 37.75 89.15, 38.27 89.95, 38.89 90.63 C 40.01 91.85, 41.47 92.67, 43.02 93.26 C 46.30 94.50, 49.99 94.68, 53.27 93.51 C 54.99 92.90, 56.61 91.91, 57.95 90.66 C 58.63 90.04, 59.23 89.35, 59.68 88.53 C 60.04 87.88, 60.30 87.15, 60.94 86.92 L 60.98 13.03 Z"/>
              <g class="hero-cortex-sulci" fill="none" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M 24.17 19.70 C 24.96 23.20, 28.38 25.18, 31.70 25.44"/>
                <path d="M 10.05 57.97 C 13.21 57.70, 17.00 57.97, 18.89 61.03"/>
                <path d="M 16.56 44.74 C 18.10 46.76, 20.84 46.81, 23.12 46.69"/>
                <path d="M 33.94 69.89 C 32.65 70.86, 31.34 71.59, 29.86 72.00 C 28.54 72.37, 27.09 72.50, 25.85 73.01 C 24.58 73.54, 23.53 74.45, 23.05 75.71 C 22.56 77.01, 22.68 78.68, 22.88 80.03"/>
                <path d="M 60.97 74.23 C 59.49 74.32, 57.82 74.86, 56.24 74.91 C 54.67 74.96, 53.19 74.52, 51.73 73.88 C 50.28 73.25, 48.85 72.43, 47.32 72.54 C 45.80 72.65, 44.18 73.67, 43.20 74.65"/>
                <path d="M 39.45 58.22 C 39.34 54.35, 42.86 51.67, 46.48 51.66"/>
                <path d="M 33.98 69.97 C 33.99 68.85, 34.13 67.70, 33.90 66.59"/>
                <path d="M 42.19 25.20 C 43.37 26.57, 45.19 27.42, 46.93 27.34 C 47.67 27.31, 48.40 27.11, 49.10 26.81 C 49.81 26.52, 50.50 26.13, 51.21 25.83 C 52.83 25.14, 54.56 24.91, 56.31 25.17 C 57.87 25.40, 59.45 26.01, 60.97 26.05"/>
                <path d="M 38.04 70.38 C 36.73 69.83, 35.10 71.01, 33.96 69.99"/>
                <path d="M 39.36 58.27 C 37.44 57.93, 35.63 57.40, 33.71 56.97 C 32.01 56.59, 30.22 56.31, 28.64 55.61 C 25.25 54.10, 22.77 50.68, 23.24 46.63"/>
                <path d="M 25.55 36.92 C 24.03 39.93, 23.39 43.35, 23.13 46.70"/>
                <path d="M 39.30 58.26 C 40.29 59.77, 42.00 60.93, 43.13 62.46 C 44.11 63.77, 44.67 65.36, 44.68 67.13"/>
              </g>
            </g>

            <g class="hero-net">
              <g class="hero-edges hero-edges-l1" stroke-linecap="round">
                <line x1="746" y1="74" x2="852" y2="96"/><line x1="746" y1="74" x2="852" y2="148"/><line x1="746" y1="74" x2="852" y2="200"/>
                <line x1="746" y1="124" x2="852" y2="96"/><line x1="746" y1="124" x2="852" y2="148"/><line x1="746" y1="124" x2="852" y2="200"/><line x1="746" y1="124" x2="852" y2="252"/>
                <line x1="746" y1="174" x2="852" y2="96"/><line x1="746" y1="174" x2="852" y2="148"/><line x1="746" y1="174" x2="852" y2="200"/><line x1="746" y1="174" x2="852" y2="252"/>
                <line x1="746" y1="224" x2="852" y2="96"/><line x1="746" y1="224" x2="852" y2="148"/><line x1="746" y1="224" x2="852" y2="200"/><line x1="746" y1="224" x2="852" y2="252"/>
                <line x1="746" y1="274" x2="852" y2="148"/><line x1="746" y1="274" x2="852" y2="200"/><line x1="746" y1="274" x2="852" y2="252"/>
              </g>
              <g class="hero-edges hero-edges-l2" stroke-linecap="round">
                <line x1="852" y1="96" x2="948" y2="122"/><line x1="852" y1="96" x2="948" y2="174"/>
                <line x1="852" y1="148" x2="948" y2="122"/><line x1="852" y1="148" x2="948" y2="174"/><line x1="852" y1="148" x2="948" y2="226"/>
                <line x1="852" y1="200" x2="948" y2="122"/><line x1="852" y1="200" x2="948" y2="174"/><line x1="852" y1="200" x2="948" y2="226"/>
                <line x1="852" y1="252" x2="948" y2="174"/><line x1="852" y1="252" x2="948" y2="226"/>
              </g>
              <g class="hero-edges hero-edges-l3" stroke-linecap="round">
                <line x1="948" y1="122" x2="1036" y2="174"/>
                <line x1="948" y1="174" x2="1036" y2="174"/>
                <line x1="948" y1="226" x2="1036" y2="174"/>
              </g>

              <g class="hero-nodes-in">
                <circle class="hn-1" cx="746" cy="74" r="8.5"/>
                <circle class="hn-2" cx="746" cy="124" r="8.5"/>
                <circle class="hn-3" cx="746" cy="174" r="8.5"/>
                <circle class="hn-4" cx="746" cy="224" r="8.5"/>
                <circle class="hn-5" cx="746" cy="274" r="8.5"/>
              </g>
              <g class="hero-nodes-mid">
                <circle cx="852" cy="96" r="8.5"/>
                <circle cx="852" cy="148" r="8.5"/>
                <circle cx="852" cy="200" r="8.5"/>
                <circle cx="852" cy="252" r="8.5"/>
                <circle cx="948" cy="122" r="8.5"/>
                <circle cx="948" cy="174" r="8.5"/>
                <circle cx="948" cy="226" r="8.5"/>
              </g>

              <circle cx="1036" cy="174" r="34" fill="url(#bd-readout-aura)" class="hero-readout-halo"/>
              <circle cx="1036" cy="174" r="21" class="hero-readout-ring"/>
              <circle cx="1036" cy="174" r="13.5" class="hero-readout-core"/>
            </g>
          </svg>

          <div class="hero-art-rail bd-mono" aria-hidden="true">
            <span><b>01</b> Raw Brain Signals</span>
            <span class="rail-arrow">→</span>
            <span><b>02</b> Clean &amp; Window</span>
            <span class="rail-arrow">→</span>
            <span><b>03</b> 103 PyTorch &amp; 48 Foundation Models</span>
            <span class="rail-arrow">→</span>
            <span class="rail-warm"><b>04</b> Decoded Brain State</span>
          </div>
        </div>

        <div class="inst-strip">
          <span class="inst-label">Maintained across</span>
          <div class="inst-logos">
            <img class="inst-ucsd" src="_static/institution_logos/ucsd.svg" alt="UC San Diego" width="205" height="39" loading="lazy" decoding="async"/>
            <img class="inst-inria" src="_static/institution_logos/inria.png" alt="Inria" width="130" height="38" loading="lazy" decoding="async"/>
            <img class="inst-freiburg" src="_static/institution_logos/unifreiburg.png" alt="University of Freiburg" width="46" height="54" loading="lazy" decoding="async"/>
            <img class="inst-donders" src="_static/institution_logos/donders.svg" alt="Donders Institute" width="132" height="44" loading="lazy" decoding="async"/>
            <img class="inst-yneuro" src="_static/institution_logos/yneuro.png" alt="Yneuro" width="114" height="40" loading="lazy" decoding="async"/>
          </div>
          <a class="inst-contrib bd-mono" href="https://github.com/braindecode/braindecode/graphs/contributors" target="_blank" rel="noopener">+ 134 contributors</a>
        </div>
      </div>
    </section>

    <section class="section hf-section" id="foundation-hub">
      <div class="container hf-split">
        <div class="hf-copy">
          <div class="hf-cobrand">
            <img class="only-light" src="_static/braindecode_long.svg" alt="Braindecode" width="164" height="28"/>
            <img class="only-dark" src="_static/braindecode_long_dark.svg" alt="Braindecode" width="164" height="28"/>
            <span class="hf-plus" aria-hidden="true">+</span>
            <img class="only-light" src="_static/ecosystem_logos/huggingface-hf-logo-with-title.svg" alt="Hugging Face" width="142" height="32"/>
            <img class="only-dark" src="_static/ecosystem_logos/huggingface-hf-logo-with-title-dark.svg" alt="Hugging Face" width="142" height="32"/>
          </div>

          <div class="hf-title-row">
            <h2 class="section-title is-left">Pretrained on the Hub</h2>
            <span class="hf-badge">141 Repos · 402K downloads/mo</span>
          </div>

          <p class="hf-desc">
            Every Braindecode model implements <code>from_pretrained()</code> and <code>push_to_hub()</code> via <code>PyTorchModelHubMixin</code>, serializing a diffable <code>safetensors + JSON</code> pair that records the Hub commit hash and pretraining corpora for auditable fine-tuning. Hub checkpoints connect to <strong>954 BIDS corpora</strong> from <code>EEGDashDataset</code> (48,000+ subjects, 138,000+ hours), while Zarr-backed datasets stream lazily via <code>pull_from_hub()</code>.
          </p>

          <div class="hf-pillars">
            <a class="hf-pillar" href="models/categorization/lbm.html">
              <span class="hf-pillar-kicker bd-mono">MODELS · 48 FOUNDATION WEIGHTS</span>
              <strong>EEGPT, LaBraM, REVE, CBraMod, SleepFM, DuIN &amp; MEGXL</strong>
              <span class="hf-pillar-sub">Readable JSON configs and parameter-level weight provenance, benchmarked under OpenEEG-Bench.</span>
            </a>
            <a class="hf-pillar" href="https://eegdash.org/dataset_summary.html" target="_blank" rel="noopener">
              <span class="hf-pillar-kicker bd-mono">DATASETS · 954 BIDS &amp; ZARR HUB CORPORA</span>
              <strong>954 BIDS datasets · 48,000+ subjects · 138,000+ hours</strong>
              <span class="hf-pillar-sub">Query EEG, MEG, iEEG, fNIRS, and sEMG cohorts via EEGDashDataset, or stream Zarr windows so pulls fetch only the chunks you read.</span>
            </a>
          </div>

          <div class="hf-actions">
            <a class="btn" href="https://huggingface.co/spaces/braindecode/OpenEEGBench" target="_blank" rel="noopener">OpenEEG-Bench</a>
            <a class="btn btn-primary" href="https://huggingface.co/braindecode" target="_blank" rel="noopener">Browse Hugging Face Hub <span aria-hidden="true">→</span></a>
          </div>
        </div>

        <div class="output-card">
          <div class="output-head">
            <span class="output-label">Output</span>
            <fieldset class="output-tabs" aria-label="Select Hugging Face Hub example">
              <label class="output-tab">
                <input type="radio" name="bd-hf-tab" value="models" class="sr-only" checked/>
                <span>Models</span>
              </label>
              <label class="output-tab">
                <input type="radio" name="bd-hf-tab" value="datasets" class="sr-only"/>
                <span>Datasets</span>
              </label>
            </fieldset>
          </div>

          <div class="output-pane output-pane-models">
            <div class="output-toolbar bd-mono">
              <span class="copy-wrap">
                <button type="button" class="output-copy-btn" data-copy="from braindecode.models import EEGPT&#10;&#10;model = EEGPT.from_pretrained(&#10;    &quot;braindecode/eegpt-pretrained&quot;,&#10;    n_outputs=4,&#10;)&#10;probs = model(X_test).softmax(dim=-1)" aria-label="Copy EEGPT from_pretrained code">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                  <span>Copy Code</span>
                </button>
                <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
              </span>
              <span>Python · safetensors</span>
            </div>
            <div class="output-terminal">
              <pre class="repl-code bd-mono" tabindex="0"><code><span class="cl"><span class="tok-kw">&gt;&gt;&gt;</span> <span class="tok-kw">from</span> braindecode.models <span class="tok-kw">import</span> <span class="tok-cls">EEGPT</span></span><span class="cl"><span class="tok-kw">&gt;&gt;&gt;</span> model = <span class="tok-cls">EEGPT</span>.<span class="tok-fn">from_pretrained</span>(</span><span class="cl">...     <span class="tok-str">"braindecode/eegpt-pretrained"</span>,</span><span class="cl">...     <span class="tok-attr">n_outputs</span>=<span class="tok-num">4</span>,</span><span class="cl">... )</span><span class="cl"><span class="tok-kw">&gt;&gt;&gt;</span> model(X_test).<span class="tok-fn">softmax</span>(<span class="tok-attr">dim</span>=-<span class="tok-num">1</span>)[:<span class="tok-num">3</span>]</span><span class="cl">tensor([[<span class="tok-num">0.92</span>, <span class="tok-num">0.03</span>, <span class="tok-num">0.03</span>, <span class="tok-num">0.02</span>],</span><span class="cl">        [<span class="tok-num">0.04</span>, <span class="tok-num">0.89</span>, <span class="tok-num">0.05</span>, <span class="tok-num">0.02</span>],</span><span class="cl">        [<span class="tok-num">0.02</span>, <span class="tok-num">0.04</span>, <span class="tok-num">0.91</span>, <span class="tok-num">0.03</span>]])</span></code></pre>
            </div>
          </div>

          <div class="output-pane output-pane-datasets">
            <div class="output-toolbar bd-mono">
              <span class="copy-wrap">
                <button type="button" class="output-copy-btn" data-copy="from eegdash import EEGDashDataset&#10;from braindecode.datasets import BaseConcatDataset&#10;&#10;bids_ds = EEGDashDataset(dataset=&quot;ds002718&quot;, subject=&quot;012&quot;)&#10;hub_ds = BaseConcatDataset.pull_from_hub(&quot;braindecode/bcic2a&quot;)&#10;X, y, crop_ind = hub_ds[0]&#10;hub_ds.push_to_hub(&quot;my-lab/motor-imagery&quot;)" aria-label="Copy EEGDash and Hub dataset code">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                  <span>Copy Code</span>
                </button>
                <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
              </span>
              <span>Python · EEGDash + Hub</span>
            </div>
            <div class="output-terminal">
              <pre class="repl-code bd-mono" tabindex="0"><code><span class="cl"><span class="tok-kw">&gt;&gt;&gt;</span> <span class="tok-kw">from</span> eegdash <span class="tok-kw">import</span> <span class="tok-cls">EEGDashDataset</span></span><span class="cl"><span class="tok-kw">&gt;&gt;&gt;</span> <span class="tok-kw">from</span> braindecode.datasets <span class="tok-kw">import</span> <span class="tok-cls">BaseConcatDataset</span></span><span class="cl"><span class="tok-kw">&gt;&gt;&gt;</span> bids_ds = <span class="tok-cls">EEGDashDataset</span>(</span><span class="cl">...     <span class="tok-attr">dataset</span>=<span class="tok-str">"ds002718"</span>, <span class="tok-attr">subject</span>=<span class="tok-str">"012"</span>,</span><span class="cl">... )</span><span class="cl"><span class="tok-kw">&gt;&gt;&gt;</span> hub_ds = <span class="tok-cls">BaseConcatDataset</span>.<span class="tok-fn">pull_from_hub</span>(</span><span class="cl">...     <span class="tok-str">"braindecode/bcic2a"</span></span><span class="cl">... )</span><span class="cl"><span class="tok-kw">&gt;&gt;&gt;</span> X, y, crop_ind = hub_ds[<span class="tok-num">0</span>]</span><span class="cl"><span class="tok-kw">&gt;&gt;&gt;</span> X.shape</span><span class="cl">(22, 1000)</span></code></pre>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section numbers-section">
      <div class="container">
        <h2 class="section-title is-center">Braindecode by the numbers</h2>

        <div class="numbers-split">
          <div class="numbers-chart-wrap">
            <div class="numbers-chart-card">
              <svg class="numbers-svg numbers-svg-fam" viewBox="0 0 540 290" role="img" aria-label="103 registered Braindecode models broken down by architecture family: 69 Convolution, 50 Attention, 48 Foundation, 23 Channel-Aware, 15 Recurrent and Graph">
                <g class="chart-grid">
                  <line x1="56" y1="36" x2="512" y2="36"/>
                  <line x1="56" y1="88" x2="512" y2="88"/>
                  <line x1="56" y1="140" x2="512" y2="140"/>
                  <line x1="56" y1="192" x2="512" y2="192"/>
                  <line x1="56" y1="226" x2="512" y2="226" class="chart-axis"/>
                </g>
                <g class="chart-yticks bd-mono" text-anchor="end">
                  <text x="46" y="40">80</text>
                  <text x="46" y="92">60</text>
                  <text x="46" y="144">40</text>
                  <text x="46" y="196">20</text>
                  <text x="46" y="230">0</text>
                </g>
                <g>
                  <rect x="82" y="62" width="44" height="164" rx="4" class="cbar-navy"/>
                  <text x="104" y="52" text-anchor="middle" class="cbar-num bd-mono">69</text>
                  <text x="104" y="248" text-anchor="middle" class="cbar-lbl bd-mono">Conv</text>

                  <rect x="172" y="107" width="44" height="119" rx="4" class="cbar-cerulean"/>
                  <text x="194" y="97" text-anchor="middle" class="cbar-num bd-mono">50</text>
                  <text x="194" y="248" text-anchor="middle" class="cbar-lbl bd-mono">Attention</text>

                  <rect x="262" y="112" width="44" height="114" rx="4" class="cbar-copper"/>
                  <text x="284" y="102" text-anchor="middle" class="cbar-num is-copper bd-mono">48</text>
                  <text x="284" y="248" text-anchor="middle" class="cbar-lbl bd-mono">Foundation</text>

                  <rect x="352" y="171" width="44" height="55" rx="4" class="cbar-cerulean"/>
                  <text x="374" y="161" text-anchor="middle" class="cbar-num bd-mono">23</text>
                  <text x="374" y="248" text-anchor="middle" class="cbar-lbl bd-mono">Montage</text>

                  <rect x="442" y="190" width="44" height="36" rx="4" class="cbar-sage"/>
                  <text x="464" y="180" text-anchor="middle" class="cbar-num bd-mono">15</text>
                  <text x="464" y="248" text-anchor="middle" class="cbar-lbl bd-mono">RNN / GNN</text>
                </g>
                <text x="284" y="276" text-anchor="middle" class="chart-caption bd-mono">103 peer-reviewed PyTorch architectures verified in CI</text>
              </svg>

              <svg class="numbers-svg numbers-svg-mod" viewBox="0 0 540 290" role="img" aria-label="Braindecode model coverage across 5 neurophysiological modalities: 77 EEG, 9 iEEG, 8 fNIRS, 7 MEG, 7 sEMG">
                <g class="chart-grid">
                  <line x1="56" y1="36" x2="512" y2="36"/>
                  <line x1="56" y1="88" x2="512" y2="88"/>
                  <line x1="56" y1="140" x2="512" y2="140"/>
                  <line x1="56" y1="192" x2="512" y2="192"/>
                  <line x1="56" y1="226" x2="512" y2="226" class="chart-axis"/>
                </g>
                <g class="chart-yticks bd-mono" text-anchor="end">
                  <text x="46" y="40">80</text>
                  <text x="46" y="92">60</text>
                  <text x="46" y="144">40</text>
                  <text x="46" y="196">20</text>
                  <text x="46" y="230">0</text>
                </g>
                <g>
                  <rect x="82" y="43" width="44" height="183" rx="4" class="cbar-navy"/>
                  <text x="104" y="33" text-anchor="middle" class="cbar-num bd-mono">77</text>
                  <text x="104" y="248" text-anchor="middle" class="cbar-lbl bd-mono">EEG / PSG</text>

                  <rect x="172" y="196" width="44" height="30" rx="4" class="cbar-sage"/>
                  <text x="194" y="186" text-anchor="middle" class="cbar-num bd-mono">9</text>
                  <text x="194" y="248" text-anchor="middle" class="cbar-lbl bd-mono">iEEG / sEEG</text>

                  <rect x="262" y="199" width="44" height="27" rx="4" class="cbar-copper"/>
                  <text x="284" y="189" text-anchor="middle" class="cbar-num is-copper bd-mono">8</text>
                  <text x="284" y="248" text-anchor="middle" class="cbar-lbl bd-mono">fNIRS</text>

                  <rect x="352" y="202" width="44" height="24" rx="4" class="cbar-cerulean"/>
                  <text x="374" y="192" text-anchor="middle" class="cbar-num bd-mono">7</text>
                  <text x="374" y="248" text-anchor="middle" class="cbar-lbl bd-mono">MEG</text>

                  <rect x="442" y="202" width="44" height="24" rx="4" class="cbar-copper"/>
                  <text x="464" y="192" text-anchor="middle" class="cbar-num is-copper bd-mono">7</text>
                  <text x="464" y="248" text-anchor="middle" class="cbar-lbl bd-mono">sEMG</text>
                </g>
                <text x="284" y="276" text-anchor="middle" class="chart-caption bd-mono">Cross-modal backbones support multiple recording types</text>
              </svg>
            </div>

            <fieldset class="numbers-legend bd-mono" aria-label="Switch chart breakdown">
              <label class="nleg-opt">
                <input type="radio" name="bd-num-mode" value="fam" class="sr-only" checked/>
                <span class="nleg-swatch is-navy"></span>
                <span>By architecture family</span>
              </label>
              <label class="nleg-opt">
                <input type="radio" name="bd-num-mode" value="mod" class="sr-only"/>
                <span class="nleg-swatch is-copper"></span>
                <span>By signal modality</span>
              </label>
            </fieldset>
          </div>

          <div class="numbers-claims">
            <div class="nclaim">
              <p class="nclaim-head">1.4M+ PyPI installs &amp; 402K monthly Hub downloads</p>
              <p class="nclaim-sub">2.1M+ installations across Braindecode, MOABB, EEGDash, and SPD Learn, maintained openly across 50+ releases and 4,600+ citations since HBM 2017.</p>
            </div>
            <div class="nclaim">
              <p class="nclaim-head">1,100+ open datasets &amp; 138,000+ hours ready to train</p>
              <p class="nclaim-sub">954 BIDS corpora across 48,000+ subjects via EEGDash and 158 BCI benchmarks via MOABB load through one dataset API.</p>
            </div>
            <div class="nclaim">
              <p class="nclaim-head">48 foundation models, checkable end to end</p>
              <p class="nclaim-sub">Change the evaluation split or preprocessing, and rankings flip. Shared protocols keep splits leak-free and baselines reproducible.</p>
            </div>
            <div>
              <a class="btn btn-primary" href="models/models_table.html">Browse the full zoo on the Models table <span aria-hidden="true">→</span></a>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="capable-band">
      <div class="container">
        <h2 class="section-title is-center is-white">Built for every neurophysiological signal</h2>

        <div class="capable-stage" data-active-cap="0">
          <div class="capable-inputs">
            <div class="cap-card" data-cap-active data-cap-mod="SCALP EEG" data-cap-sig="X: (B, 64, 1000) @ 250 Hz → n_chans, n_times, sfreq derived">
              <div class="cap-card-top">
                <h3>Scalp EEG &amp; Sleep PSG</h3>
                <span class="cap-count bd-mono">77 models</span>
              </div>
              <p class="cap-stream bd-mono">19–128 ch @ 128–512 Hz · C3 Cz C4 Fpz-Cz Pz-Oz · MNE Raw / BIDS / EDF</p>
              <div class="cap-links bd-mono">
                <a href="generated/braindecode.models.EEGPT.html">EEGPT</a>
                <a href="generated/braindecode.models.Labram.html">LaBraM</a>
                <a href="generated/braindecode.models.REVE.html">REVE</a>
                <a href="generated/braindecode.models.EEGNeX.html">EEGNeX</a>
                <a href="generated/braindecode.models.SleepFM.html">SleepFM</a>
              </div>
            </div>

            <div class="cap-card" data-cap-mod="iEEG · MEG" data-cap-sig="X: (B, 128, 2048) @ 1 kHz → chs_info 3D coords derived">
              <div class="cap-card-top">
                <h3>Intracranial iEEG, ECoG &amp; MEG</h3>
                <span class="cap-count bd-mono">16 models</span>
              </div>
              <p class="cap-stream bd-mono">64–306 sensors @ 500–2048 Hz · 3D stereotactic shafts &amp; OPM/SQUID arrays</p>
              <div class="cap-links bd-mono">
                <a href="generated/braindecode.models.DuIN.html">DuIN</a>
                <a href="generated/braindecode.models.PopulationTransformer.html">PopT</a>
                <a href="generated/braindecode.models.BrainBERT.html">BrainBERT</a>
                <a href="generated/braindecode.models.MEGXL.html">MEGXL</a>
                <a href="generated/braindecode.models.BrainOmni.html">BrainOmni</a>
              </div>
            </div>

            <div class="cap-card" data-cap-mod="fNIRS · sEMG" data-cap-sig="X: (B, 48, 150) @ 10 Hz → input_window_seconds derived">
              <div class="cap-card-top">
                <h3>fNIRS Optical &amp; Surface sEMG</h3>
                <span class="cap-count bd-mono">15 models</span>
              </div>
              <p class="cap-stream bd-mono">HbO/HbR optode pairs @ 10 Hz · 2 kHz motor-unit wrist arrays</p>
              <div class="cap-links bd-mono">
                <a href="generated/braindecode.models.GeoNIRS.html">GeoNIRS</a>
                <a href="generated/braindecode.models.MtReNIRS.html">MtReNIRS</a>
                <a href="generated/braindecode.models.TinyMyo.html">TinyMyo</a>
                <a href="generated/braindecode.models.MetaNeuromotorHand.html">MetaNeuromotor</a>
                <a href="generated/braindecode.models.VEMG2Pose.html">VEMG2Pose</a>
              </div>
            </div>
          </div>

          <div class="capable-curves" aria-hidden="true">
            <svg viewBox="0 0 180 396" fill="none" preserveAspectRatio="none">
              <defs>
                <linearGradient id="bd-cap-pulse-grad" x1="6" y1="0" x2="174" y2="0" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#ffffff" stop-opacity="0.22"/>
                  <stop offset="52%" stop-color="#f8c697" stop-opacity="0.96"/>
                  <stop offset="100%" stop-color="#ffffff" stop-opacity="1"/>
                </linearGradient>
                <filter id="bd-cap-glow" x="-25%" y="-25%" width="150%" height="150%">
                  <feGaussianBlur stdDeviation="2.6" result="blur"/>
                  <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>

              <path class="cap-base-path b0" d="M6 66 C94 66, 88 198, 174 198"/>
              <path class="cap-base-path b1" d="M6 198 L174 198"/>
              <path class="cap-base-path b2" d="M6 330 C94 330, 88 198, 174 198"/>
              <path class="cap-stem-path" d="M174 198 L180 198"/>

              <g class="cap-pulse-group g0" filter="url(#bd-cap-glow)">
                <path class="cap-pulse-path p1" pathLength="100" d="M6 66 C94 66, 88 198, 174 198"/>
                <circle class="cap-orb o1" cx="0" cy="0" r="3.6"/>
              </g>
              <g class="cap-pulse-group g1" filter="url(#bd-cap-glow)">
                <path class="cap-pulse-path p2" pathLength="100" d="M6 198 L174 198"/>
                <circle class="cap-orb o2" cx="0" cy="0" r="3.6"/>
              </g>
              <g class="cap-pulse-group g2" filter="url(#bd-cap-glow)">
                <path class="cap-pulse-path p3" pathLength="100" d="M6 330 C94 330, 88 198, 174 198"/>
                <circle class="cap-orb o3" cx="0" cy="0" r="3.6"/>
              </g>

              <circle class="cap-node n0" cx="6" cy="66" r="4.2"/>
              <circle class="cap-node n1" cx="6" cy="198" r="4.2"/>
              <circle class="cap-node n2" cx="6" cy="330" r="4.2"/>

              <circle class="cap-junction-halo" cx="174" cy="198" r="10"/>
              <circle class="cap-junction-node" cx="174" cy="198" r="4.8"/>
            </svg>
          </div>

          <div class="capable-out">
            <div class="cap-out-top">
              <span class="cap-pill bd-mono">SHARED BASE CLASS · ALL 103 MODELS</span>
              <a class="cap-doc-link bd-mono" href="generated/braindecode.models.EEGModuleMixin.html">Docs →</a>
            </div>

            <div class="cap-mixin-head">
              <h3 class="cap-mixin-title bd-mono">
                <a href="generated/braindecode.models.EEGModuleMixin.html">EEGModuleMixin</a>
              </h3>
              <p class="cap-mixin-bases bd-mono">+ PyTorchModelHubMixin · torch.nn.Module</p>
            </div>

            <div class="cap-live-sig bd-mono" aria-live="off">
              <span class="cap-live-dot" aria-hidden="true"></span>
              <span class="cap-live-mod" data-cap-live-mod>SCALP EEG</span>
              <span class="cap-live-sep" aria-hidden="true">·</span>
              <code class="cap-live-expr" data-cap-live-expr>X: (B, 64, 1000) @ 250 Hz → n_chans, n_times, sfreq derived</code>
            </div>

            <div class="cap-contract-grid">
              <div class="cap-contract-item">
                <strong class="cap-contract-title bd-mono">Auto-sized geometry</strong>
                <p>Pass any subset of <code>n_chans</code>, <code>n_times</code>, <code>sfreq</code>, <code>input_window_seconds</code>, or <code>chs_info</code>; missing shapes are derived automatically.</p>
              </div>
              <div class="cap-contract-item">
                <strong class="cap-contract-title bd-mono">Hub safetensors</strong>
                <p><code>from_pretrained()</code> and <code>push_to_hub()</code> with JSON config and parameter-level provenance.</p>
              </div>
              <div class="cap-contract-item">
                <strong class="cap-contract-title bd-mono">Transfer &amp; probing</strong>
                <p><code>reset_head(n_outputs)</code> and <code>return_features=True</code> across classifiers and foundation models.</p>
              </div>
              <div class="cap-contract-item">
                <strong class="cap-contract-title bd-mono">Zero lock-in</strong>
                <p>Standard <code>torch.nn.Module</code> compatible with <code>EEGClassifier</code>, Lightning, and <code>torch.compile</code>.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section venue-section venue-showcase">
      <div class="container">
        <div class="section-head-stack is-center">
          <span class="section-eyebrow">USED BY THE COMMUNITY</span>
          <h2 class="section-title">What labs and students build with Braindecode</h2>
          <p class="section-sub">From student thesis projects to hospital cohorts and NeurIPS competitions, researchers use Braindecode to run checkable experiments and share their models back.</p>
        </div>
      </div>

      <div class="venue-track" tabindex="0" aria-label="Scientific venues and research highlights powered by Braindecode">
        <a class="venue-card tone-navy" href="https://neural-interfaces26.github.io/" target="_blank" rel="noopener">
          <svg class="venue-art" viewBox="0 0 396 432" fill="none" aria-hidden="true">
            <circle cx="320" cy="108" r="138" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
            <circle cx="320" cy="108" r="96" stroke="rgba(255,255,255,0.14)" stroke-width="1"/>
            <circle cx="320" cy="108" r="56" stroke="rgba(242,176,119,0.34)" stroke-width="1.2"/>
            <path d="M0 192 Q50 158 100 192 T200 192 T300 192 T400 192" stroke="rgba(137,200,247,0.36)" stroke-width="1.6"/>
          </svg>
          <div class="venue-brand-head">
            <img class="venue-logo is-tall dark-light" src="_static/venue_logos/neurips-lockup-white.svg" alt="NeurIPS" width="142" height="42" loading="lazy" decoding="async"/>
            <span class="venue-subtag bd-mono">COMPETITIONS</span>
          </div>
          <div class="venue-body">
            <p class="venue-quote">1,197 teams from 247 institutions competed in the NeurIPS 2025 EEG Foundation Challenge on Braindecode and EEGDash, returning in 2026 across 4 EEG &amp; EMG tracks.</p>
            <p class="venue-cite">NeurIPS 2026 EEG/EMG Challenge · NeurIPS 2025 Challenge · BEETL 2021</p>
            <span class="venue-cta bd-mono">Explore NeurIPS challenges →</span>
          </div>
        </a>

        <a class="venue-card tone-copper" href="https://doi.org/10.1016/j.ebiom.2024.105259" target="_blank" rel="noopener">
          <svg class="venue-art" viewBox="0 0 396 432" fill="none" aria-hidden="true">
            <path d="M10 148 C76 74, 142 220, 208 136 C274 62, 332 204, 388 130" stroke="rgba(245,194,147,0.4)" stroke-width="1.8"/>
            <path d="M10 186 C76 112, 142 258, 208 174 C274 100, 332 242, 388 168" stroke="rgba(255,255,255,0.16)" stroke-width="1.2"/>
          </svg>
          <div class="venue-brand-head">
            <img class="venue-logo is-compact dark-light" src="_static/venue_logos/thelancet-white.svg" alt="The Lancet" width="196" height="24" loading="lazy" decoding="async"/>
            <span class="venue-subtag bd-mono">eBIOMEDICINE</span>
          </div>
          <div class="venue-body">
            <p class="venue-quote">Machine learning of brain-specific wavelet biomarkers from clinical EEG across neurological cohorts.</p>
            <p class="venue-cite">Bomatter, Paillard, Garces, Hipp &amp; Engemann · F. Hoffmann-La Roche (2024)</p>
            <span class="venue-cta bd-mono">Read clinical study →</span>
          </div>
        </a>

        <a class="venue-card tone-invite" href="https://github.com/braindecode/braindecode/discussions" target="_blank" rel="noopener">
          <svg class="venue-art" viewBox="0 0 396 432" fill="none" aria-hidden="true">
            <circle cx="312" cy="96" r="96" stroke="rgba(243,178,120,0.28)" stroke-dasharray="5 5" stroke-width="1.4"/>
            <path d="M24 188 Q86 154 148 188 T272 188 T396 188" stroke="rgba(243,178,120,0.34)" stroke-width="1.6"/>
          </svg>
          <div class="venue-brand-head">
            <img class="venue-logo is-compact dark-light" src="_static/braindecode_long_dark.svg" alt="Braindecode" width="168" height="27" loading="lazy" decoding="async"/>
            <span class="venue-subtag is-invite bd-mono">COMMUNITY</span>
          </div>
          <div class="venue-body">
            <p class="venue-quote venue-invite-title">Your next paper here :)</p>
            <p class="venue-cite">Working on an EEG, iEEG, MEG, fNIRS, or sEMG study with Braindecode? Tell us about your paper, push weights to Hugging Face, or add your architecture to the zoo.</p>
            <span class="venue-cta bd-mono">Share your work with us →</span>
          </div>
        </a>

        <a class="venue-card tone-cerulean" href="https://openreview.net/forum?id=5W7n3aXjGP" target="_blank" rel="noopener">
          <svg class="venue-art" viewBox="0 0 396 432" fill="none" aria-hidden="true">
            <path d="M18 176 Q68 124 118 176 T218 176 T318 176 T396 176" stroke="rgba(242,176,119,0.36)" stroke-width="1.6"/>
            <path d="M18 212 Q68 164 118 212 T218 212 T318 212 T396 212" stroke="rgba(137,200,247,0.28)" stroke-width="1.3"/>
          </svg>
          <div class="venue-brand-head">
            <img class="venue-logo is-tall dark-light" src="_static/venue_logos/neurips-lockup-white.svg" alt="NeurIPS" width="142" height="42" loading="lazy" decoding="async"/>
            <span class="venue-subtag bd-mono">METHODS</span>
          </div>
          <div class="venue-body">
            <p class="venue-quote">Convolutional Monge-mapping normalization for sleep EEG, differentiable invariance layers, and multi-scale wavelet pretraining run on Braindecode.</p>
            <p class="venue-cite">CMMN (Gnassounou et al., 2023) · AugNet (Rommel et al., 2022) · PhysioWave (Chen et al., 2025)</p>
            <span class="venue-cta bd-mono">Read NeurIPS paper →</span>
          </div>
        </a>

        <a class="venue-card tone-sage" href="https://doi.org/10.1038/s42003-025-08464-3" target="_blank" rel="noopener">
          <svg class="venue-art" viewBox="0 0 396 432" fill="none" aria-hidden="true">
            <rect x="44" y="106" width="308" height="114" rx="8" stroke="rgba(255,255,255,0.14)" stroke-dasharray="4 4"/>
            <path d="M24 162 L98 162 L114 112 L128 206 L144 140 L160 178 L178 162 L372 162" stroke="rgba(155,224,171,0.48)" stroke-width="1.8"/>
          </svg>
          <div class="venue-brand-head">
            <img class="venue-logo is-compact dark-light" src="_static/venue_logos/nature-white.svg" alt="Nature Portfolio" width="128" height="32" loading="lazy" decoding="async"/>
            <span class="venue-subtag bd-mono">COMMUNICATIONS BIOLOGY</span>
          </div>
          <div class="venue-body">
            <p class="venue-quote">Multiverse analysis across 2,592 preprocessing paths and seven ERP CORE paradigms using Braindecode to measure how filtering and artifact removal shape decoding.</p>
            <p class="venue-cite">Kessler, Enge &amp; Skeide · Max Planck CBS (Communications Biology, 2025)</p>
            <span class="venue-cta bd-mono">Read preprocessing study →</span>
          </div>
        </a>

        <a class="venue-card tone-navy" href="https://openreview.net/forum?id=rWkk4373xa" target="_blank" rel="noopener">
          <svg class="venue-art" viewBox="0 0 396 432" fill="none" aria-hidden="true">
            <g stroke="rgba(255,255,255,0.15)" stroke-width="1">
              <line x1="40" y1="84" x2="344" y2="204"/>
              <line x1="40" y1="144" x2="344" y2="204"/>
              <line x1="40" y1="204" x2="344" y2="124"/>
              <line x1="40" y1="144" x2="344" y2="124"/>
            </g>
            <circle cx="344" cy="124" r="8" fill="rgba(242,176,119,0.55)"/>
            <circle cx="344" cy="204" r="10" fill="rgba(137,200,247,0.6)"/>
          </svg>
          <div class="venue-brand-head">
            <img class="venue-logo dark-light" src="_static/venue_logos/iclr-white.svg" alt="ICLR" width="136" height="38" loading="lazy" decoding="async"/>
            <span class="venue-subtag bd-mono">AUGMENTATION</span>
          </div>
          <div class="venue-body">
            <p class="venue-quote">Class-wise automatic differentiable data augmentation for sleep staging and motor-imagery EEG, contributed directly to braindecode.augmentation.</p>
            <p class="venue-cite">CADDA · Rommel, Moreau, Paillard &amp; Gramfort (ICLR 2022)</p>
            <span class="venue-cta bd-mono">Read ICLR paper →</span>
          </div>
        </a>

        <a class="venue-card tone-slate" href="https://proceedings.mlr.press/v306/xiong26f.html" target="_blank" rel="noopener">
          <svg class="venue-art" viewBox="0 0 396 432" fill="none" aria-hidden="true">
            <rect x="64" y="154" width="34" height="76" rx="3" fill="rgba(255,255,255,0.1)"/>
            <rect x="126" y="126" width="34" height="104" rx="3" fill="rgba(255,255,255,0.16)"/>
            <rect x="188" y="98" width="34" height="132" rx="3" fill="rgba(137,200,247,0.32)"/>
            <rect x="250" y="72" width="34" height="158" rx="3" fill="rgba(242,176,119,0.44)"/>
          </svg>
          <div class="venue-brand-head">
            <img class="venue-logo dark-light" src="_static/venue_logos/icml-white.svg" alt="ICML" width="116" height="38" loading="lazy" decoding="async"/>
            <span class="venue-subtag bd-mono">FM BENCHMARKS</span>
          </div>
          <div class="venue-body">
            <p class="venue-quote">Standardized evaluation and diagnostic analysis of EEG foundation models and Braindecode baselines across 14 datasets and 10 paradigms.</p>
            <p class="venue-cite">EEG-FM-Bench · Xiong, Li, Li, Zhu &amp; Jiang (ICML 2026, PMLR v306)</p>
            <span class="venue-cta bd-mono">Read ICML benchmark →</span>
          </div>
        </a>

        <a class="venue-card tone-cerulean" href="https://github.com/facebookresearch/neuroai" target="_blank" rel="noopener">
          <svg class="venue-art" viewBox="0 0 396 432" fill="none" aria-hidden="true">
            <ellipse cx="198" cy="152" rx="124" ry="54" stroke="rgba(137,200,247,0.28)" stroke-width="1.4"/>
            <ellipse cx="198" cy="152" rx="124" ry="54" transform="rotate(26 198 152)" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
            <ellipse cx="198" cy="152" rx="124" ry="54" transform="rotate(-26 198 152)" stroke="rgba(242,176,119,0.26)" stroke-width="1.2"/>
          </svg>
          <div class="venue-brand-head">
            <img class="venue-logo is-compact dark-light" src="_static/venue_logos/meta-brainai-white.png" alt="Meta Brain &amp; AI" width="156" height="33" loading="lazy" decoding="async"/>
            <span class="venue-subtag bd-mono">BRAIN &amp; AI</span>
          </div>
          <div class="venue-body">
            <p class="venue-quote">Meta Brain &amp; AI co-leads tracks in the NeurIPS 2026 EEG/EMG Challenge and builds NeuralTrain and NeuralBench directly on Braindecode.</p>
            <p class="venue-cite">Meta Brain &amp; AI · facebookresearch/neuroai · NeurIPS 2026 Challenge</p>
            <span class="venue-cta bd-mono">Explore NeuroAI suite →</span>
          </div>
        </a>

        <a class="venue-card tone-copper" href="https://github.com/scikit-adaptation/skada-bench" target="_blank" rel="noopener">
          <svg class="venue-art" viewBox="0 0 396 432" fill="none" aria-hidden="true">
            <circle cx="110" cy="150" r="48" stroke="rgba(255,255,255,0.16)" stroke-width="1.2"/>
            <circle cx="200" cy="150" r="48" stroke="rgba(137,200,247,0.34)" stroke-width="1.4"/>
            <circle cx="290" cy="150" r="48" stroke="rgba(242,176,119,0.38)" stroke-width="1.4"/>
          </svg>
          <div class="venue-brand-head">
            <img class="venue-logo is-compact dark-light" src="_static/venue_logos/tmlr-white.svg" alt="TMLR" width="128" height="32" loading="lazy" decoding="async"/>
            <span class="venue-subtag bd-mono">+ CVPR 2026</span>
          </div>
          <div class="venue-body">
            <p class="venue-quote">Realistic unsupervised domain-adaptation validation across cross-subject BCI cohorts and test-time calibration for EEG-to-image retrieval.</p>
            <p class="venue-cite">SKADA-Bench (Lalou, Gnassounou et al., TMLR) · SATTC (Huang &amp; Zhu, CVPR 2026)</p>
            <span class="venue-cta bd-mono">Explore benchmark code →</span>
          </div>
        </a>
      </div>

      <div class="venue-controls" aria-label="Carousel pagination">
        <button type="button" class="venue-arrow" data-venue-prev aria-label="Previous highlight">‹</button>
        <div class="venue-dots" role="tablist">
          <button type="button" class="venue-dot" role="tab" aria-selected="true" data-active aria-label="Highlight 1"></button>
          <button type="button" class="venue-dot" role="tab" aria-selected="false" aria-label="Highlight 2"></button>
          <button type="button" class="venue-dot" role="tab" aria-selected="false" aria-label="Highlight 3"></button>
          <button type="button" class="venue-dot" role="tab" aria-selected="false" aria-label="Highlight 4"></button>
          <button type="button" class="venue-dot" role="tab" aria-selected="false" aria-label="Highlight 5"></button>
          <button type="button" class="venue-dot" role="tab" aria-selected="false" aria-label="Highlight 6"></button>
          <button type="button" class="venue-dot" role="tab" aria-selected="false" aria-label="Highlight 7"></button>
          <button type="button" class="venue-dot" role="tab" aria-selected="false" aria-label="Highlight 8"></button>
          <button type="button" class="venue-dot" role="tab" aria-selected="false" aria-label="Highlight 9"></button>
        </div>
        <button type="button" class="venue-arrow" data-venue-next aria-label="Next highlight">›</button>
      </div>
    </section>

    <section class="section core-section" id="core-components">
      <div class="container">
        <div class="core-intro-split">
          <div class="core-intro-copy">
            <span class="section-eyebrow">WHY BRAINDECODE · CORE STACK</span>
            <h2 class="section-title">Everything you need to go from signal to prediction</h2>
            <p class="section-sub">
              Built to plug into the neurophysiology ecosystem you already use: every
              <a href="https://moabb.neurotechx.com/" target="_blank" rel="noopener">MOABB</a> benchmark (158 datasets), every
              <a href="https://mne.tools/" target="_blank" rel="noopener">MNE-Python</a> and
              <a href="https://sccn.github.io/eegprep/" target="_blank" rel="noopener">EEGPrep</a> preprocessing function,
              <a href="https://facebookresearch.github.io/neuroai/neuralbench/" target="_blank" rel="noopener">NeuralBench</a> tasks,
              <a href="https://spdlearn.org/" target="_blank" rel="noopener">SPD Learn</a>, and 954 BIDS corpora via
              <a href="https://eegdash.org/" target="_blank" rel="noopener">EEGDash</a>. One library, no lock-in.
            </p>
          </div>

          <div class="core-glance-card" aria-label="Braindecode architecture at a glance">
            <div class="core-glance-head bd-mono">
              <span>ONE LIBRARY · NO LOCK-IN</span>
              <a href="api.html">API docs →</a>
            </div>
            <div class="core-glance-rows">
              <a class="core-glance-row" href="api.html#datasets">
                <span class="cg-idx bd-mono">01</span>
                <span class="cg-stage">Ingest &amp; Clean</span>
                <span class="cg-spec bd-mono">MNE Raw/Epochs · BIDS · EEGDash · MOABB · EEGPrep</span>
              </a>
              <a class="core-glance-row" href="models/models_table.html">
                <span class="cg-idx bd-mono">02</span>
                <span class="cg-stage">Model &amp; Adapt</span>
                <span class="cg-spec bd-mono">103 nn.Module · 48 Hub weights · chs_info</span>
              </a>
              <a class="core-glance-row" href="api.html#training">
                <span class="cg-idx bd-mono">03</span>
                <span class="cg-stage">Train &amp; Audit</span>
                <span class="cg-spec bd-mono">sklearn fit() · Lightning · torch.compile · SzCORE</span>
              </a>
            </div>
          </div>
        </div>

        <div class="core-why-grid">
          <a class="core-why-cell" href="models/models_table.html">
            <div class="core-why-top">
              <span class="core-why-icon" aria-hidden="true">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0 4 4 2 0"/></svg>
              </span>
              <h3>Decode raw electrophysiology</h3>
              <p>End-to-end models go straight from raw EEG, iEEG, MEG, fNIRS, and sEMG to predictions. No hand-crafted features required.</p>
            </div>
            <span class="core-why-link">103 models <span aria-hidden="true">→</span></span>
          </a>

          <a class="core-why-cell" href="api.html">
            <div class="core-why-top">
              <span class="core-why-icon" aria-hidden="true">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v6m0 8v6M4.93 4.93l4.24 4.24m5.66 5.66 4.24 4.24M2 12h6m8 0h6M4.93 19.07l4.24-4.24m5.66-5.66 4.24-4.24"/></svg>
              </span>
              <h3>PyTorch-native, sklearn-friendly</h3>
              <p>Built on PyTorch and skorch with a Lightning-friendly API. Use <strong>EEGClassifier</strong> and <strong>EEGRegressor</strong> for cross-validation, or drop any <code>nn.Module</code> into <code>torch.compile</code> and your own loop.</p>
            </div>
            <span class="core-why-link">API reference <span aria-hidden="true">→</span></span>
          </a>

          <a class="core-why-cell" href="api.html#datasets">
            <div class="core-why-top">
              <span class="core-why-icon" aria-hidden="true">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/></svg>
              </span>
              <h3>Every MOABB, NeuralBench &amp; BIDS dataset</h3>
              <p>All <strong>158 MOABB datasets</strong> work out of the box with <code>MOABBDataset</code>, plus <strong>NeuralBench</strong> tasks, TUH, CHB-MIT, Sleep Physionet, and <strong>954 BIDS corpora</strong> (48,000+ subjects) via EEGDash.</p>
            </div>
            <span class="core-why-link">Datasets <span aria-hidden="true">→</span></span>
          </a>

          <a class="core-why-cell" href="api.html#preprocessing">
            <div class="core-why-top">
              <span class="core-why-icon" aria-hidden="true">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.5 10 19 14 21 14 12.5 22 3"/></svg>
              </span>
              <h3>Preprocessing &amp; augmentations</h3>
              <p>Compatible with every MNE preprocessing function and automated <strong>EEGPrep</strong> ASR cleaning, plus exponential standardization and 20+ differentiable EEG augmentations.</p>
            </div>
            <span class="core-why-link">Preprocessing &amp; augmentation <span aria-hidden="true">→</span></span>
          </a>

          <a class="core-why-cell" href="models/channel_strategies.html">
            <div class="core-why-top">
              <span class="core-why-icon" aria-hidden="true">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2.2"/><circle cx="6" cy="18" r="2.2"/><circle cx="18" cy="12" r="2.2"/><path d="M8.2 6.8 15.8 11M8.2 17.2 15.8 13"/></svg>
              </span>
              <h3>Heterogeneous electrode montages</h3>
              <p>Train position-aware models across mixed caps with <code>chs_info</code> and <code>pad_channels_collate</code>, or select benchmarked <code>ChannelLayer</code> adapters when deploying rigid checkpoints (Kokate et al., 2026).</p>
            </div>
            <span class="core-why-link">Channel adaptation <span aria-hidden="true">→</span></span>
          </a>

          <a class="core-why-cell" href="auto_examples/datasets_io/plot_mne_dataset_example.html">
            <div class="core-why-top">
              <span class="core-why-icon" aria-hidden="true">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 0 7 4.5v.5a2.5 2.5 0 0 0-2 2.4V8a2.5 2.5 0 0 0-2 2.4v.6a2.5 2.5 0 0 0 1 2 2.5 2.5 0 0 0-1 2v.6a2.5 2.5 0 0 0 2 2.4V19a2.5 2.5 0 0 0 2.5 2.5A2.5 2.5 0 0 0 10 19V4.5A2.5 2.5 0 0 0 7.5 2h2zM14.5 2A2.5 2.5 0 0 1 17 4.5v.5a2.5 2.5 0 0 1 2 2.4V8a2.5 2.5 0 0 1 2 2.4v.6a2.5 2.5 0 0 1-1 2 2.5 2.5 0 0 1 1 2v.6a2.5 2.5 0 0 1-2 2.4V19a2.5 2.5 0 0 1-2.5 2.5A2.5 2.5 0 0 1 14 19V4.5"/></svg>
              </span>
              <h3>Plays well with MNE-Python</h3>
              <p>Native interop with <code>mne.io.Raw</code> and <code>mne.Epochs</code>. Reuse the sensor montages and analysis tools you already know for QC and visualization.</p>
            </div>
            <span class="core-why-link">MNE workflow <span aria-hidden="true">→</span></span>
          </a>
        </div>

        <div class="pipeline-subhead">
          <div>
            <h3>A unified API, nine modules</h3>
            <p>From raw recording to trained checkpoint, the same nine modules cover every task: classification, regression, sleep staging, seizure detection, and self-supervised pretraining.</p>
          </div>
          <a class="btn btn-sm" href="api.html">Full API reference <span aria-hidden="true">→</span></a>
        </div>

        <div class="pipeline-grid">
          <div class="pipeline-col">
            <div class="pipeline-col-head">
              <span class="pipeline-phase">PHASE 01</span>
              <h3 class="pipeline-col-title">Data &amp; Cleaning</h3>
            </div>
            <a class="core-card" href="api.html#datasets">
              <span class="core-card-icon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/></svg></span>
              <span class="core-card-body">
                <span class="core-card-name">braindecode.datasets</span>
                <span class="core-card-desc">MOABB, BIDS, EEGDash (954 corpora), Hub datasets, TUH &amp; Sleep.</span>
              </span>
            </a>
            <a class="core-card" href="api.html#preprocessing">
              <span class="core-card-icon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="22 3 2 3 10 12.5 10 19 14 21 14 12.5 22 3"/></svg></span>
              <span class="core-card-body">
                <span class="core-card-name">braindecode.preprocessing</span>
                <span class="core-card-desc">MNE-backed filtering, automated EEGPrep ASR, trial &amp; fixed windowing.</span>
              </span>
            </a>
            <a class="core-card" href="api.html#samplers">
              <span class="core-card-icon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5"/></svg></span>
              <span class="core-card-body">
                <span class="core-card-name">braindecode.samplers</span>
                <span class="core-card-desc">Balanced trial, recording, and sequence samplers for SSL &amp; sleep.</span>
              </span>
            </a>
          </div>

          <div class="pipeline-col">
            <div class="pipeline-col-head">
              <span class="pipeline-phase">PHASE 02</span>
              <h3 class="pipeline-col-title">Modeling &amp; Geometry</h3>
            </div>
            <a class="core-card" href="models/models_table.html">
              <span class="core-card-icon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M8 6h2a3 3 0 0 1 3 3 3 3 0 0 0 3 3M8 18h2a3 3 0 0 0 3-3 3 3 0 0 1 3-3"/></svg></span>
              <span class="core-card-body">
                <span class="core-card-name">braindecode.models</span>
                <span class="core-card-desc">103 architectures &amp; 48 foundation backbones via <code>EEGModuleMixin</code>.</span>
              </span>
            </a>
            <a class="core-card" href="api.html#modules">
              <span class="core-card-icon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m12 2 9 5-9 5-9-5 9-5z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></svg></span>
              <span class="core-card-body">
                <span class="core-card-name">braindecode.modules</span>
                <span class="core-card-desc">Patch tokenizers, RVQ codebooks, filterbanks, spatial kernels &amp; channel layers.</span>
              </span>
            </a>
            <a class="core-card" href="api.html#augmentation">
              <span class="core-card-icon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2v6m0 8v6M4.93 4.93l4.24 4.24m5.66 5.66 4.24 4.24M2 12h6m8 0h6"/></svg></span>
              <span class="core-card-body">
                <span class="core-card-name">braindecode.augmentation</span>
                <span class="core-card-desc">20+ differentiable time, spectral, and sensor-rotation augmentations.</span>
              </span>
            </a>
          </div>

          <div class="pipeline-col">
            <div class="pipeline-col-head">
              <span class="pipeline-phase">PHASE 03</span>
              <h3 class="pipeline-col-title">Train, Score &amp; Interpret</h3>
            </div>
            <a class="core-card" href="api.html#classifier">
              <span class="core-card-icon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 4 20 12 6 20 6 4"/></svg></span>
              <span class="core-card-body">
                <span class="core-card-name">EEGClassifier / EEGRegressor</span>
                <span class="core-card-desc">skorch wrappers with scikit-learn <code>.fit()</code>, <code>.predict_proba()</code> &amp; CV.</span>
              </span>
            </a>
            <a class="core-card" href="api.html#training">
              <span class="core-card-icon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 6 9 17l-5-5"/></svg></span>
              <span class="core-card-body">
                <span class="core-card-name">braindecode.training</span>
                <span class="core-card-desc">Cropped trial decoding, clinical SzCORE seizure metrics &amp; callbacks.</span>
              </span>
            </a>
            <a class="core-card" href="api.html#visualization">
              <span class="core-card-icon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 12h4l3-9 4 18 3-9h4"/></svg></span>
              <span class="core-card-body">
                <span class="core-card-name">braindecode.visualization</span>
                <span class="core-card-desc">Spectral input-gradient maps, SAE probes &amp; MNE scalp topomaps.</span>
              </span>
            </a>
          </div>
        </div>
      </div>
    </section>

    <section class="qs-band qs-section" id="quickstart">
      <div class="container">
        <div class="section-head-stack is-center">
          <span class="section-eyebrow">QUICKSTART · END-TO-END PIPELINE</span>
          <h2 class="section-title is-white">Train a model in 15 lines</h2>
          <p class="section-sub">From raw EEG to a trained decoder. Each step is a stand-alone module; swap any piece without touching the others, or switch tabs for Hub fine-tuning and clinical SzCORE evaluation.</p>
        </div>

        <div class="qs-stage">
          <div class="qs-steps-col">
            <fieldset class="qs-steps qs-steps-moabb" aria-label="Supervised benchmark pipeline steps">
              <label class="qs-step">
                <input type="radio" name="bd-qstep-moabb" value="1" class="sr-only" checked/>
                <span class="qs-step-num">01</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Load an open BCI benchmark</span>
                  <span class="qs-step-desc">Fetch BCI Competition IV-2a via MOABB, query 954 BIDS corpora from EEGDash, or wrap MNE Raw files.</span>
                </span>
              </label>

              <label class="qs-step">
                <input type="radio" name="bd-qstep-moabb" value="2" class="sr-only"/>
                <span class="qs-step-num">02</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Clean, filter &amp; slice windows</span>
                  <span class="qs-step-desc">Apply reproducible MNE or EEGPrep preprocessors, then cut event-locked trial windows.</span>
                </span>
              </label>

              <label class="qs-step">
                <input type="radio" name="bd-qstep-moabb" value="3" class="sr-only"/>
                <span class="qs-step-num">03</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Pick any of 103 architectures</span>
                  <span class="qs-step-desc">Channel count, window samples, and sampling rate are inferred automatically from dataset metadata.</span>
                </span>
              </label>

              <label class="qs-step">
                <input type="radio" name="bd-qstep-moabb" value="4" class="sr-only"/>
                <span class="qs-step-num">04</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Train &amp; cross-validate anywhere</span>
                  <span class="qs-step-desc">Run scikit-learn fit/score via skorch or pass windows directly to PyTorch and Lightning loops.</span>
                </span>
              </label>
            </fieldset>

            <fieldset class="qs-steps qs-steps-montage" aria-label="Pretrained Hub fine-tuning steps">
              <label class="qs-step">
                <input type="radio" name="bd-qstep-montage" value="1" class="sr-only" checked/>
                <span class="qs-step-num">01</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Read recording &amp; sensor metadata</span>
                  <span class="qs-step-desc">Load your MNE recording with its native electrode names and 3D montage coordinates.</span>
                </span>
              </label>

              <label class="qs-step">
                <input type="radio" name="bd-qstep-montage" value="2" class="sr-only"/>
                <span class="qs-step-num">02</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Pull audited weights from Hugging Face</span>
                  <span class="qs-step-desc">Download safetensors foundation weights and commit provenance in a single call.</span>
                </span>
              </label>

              <label class="qs-step">
                <input type="radio" name="bd-qstep-montage" value="3" class="sr-only"/>
                <span class="qs-step-num">03</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Configure classes &amp; montage handling</span>
                  <span class="qs-step-desc">Set n_outputs for your task, keeping native channels for flexible models or adapting rigid caps.</span>
                </span>
              </label>

              <label class="qs-step">
                <input type="radio" name="bd-qstep-montage" value="4" class="sr-only"/>
                <span class="qs-step-num">04</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Probe features or fine-tune</span>
                  <span class="qs-step-desc">Wrap the pretrained backbone in EEGClassifier to probe frozen embeddings or fine-tune end to end.</span>
                </span>
              </label>
            </fieldset>

            <fieldset class="qs-steps qs-steps-szcore" aria-label="Clinical SzCORE seizure evaluation steps">
              <label class="qs-step">
                <input type="radio" name="bd-qstep-szcore" value="1" class="sr-only" checked/>
                <span class="qs-step-num">01</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Slice patient-disjoint clinical EEG</span>
                  <span class="qs-step-desc">Cut continuous CHB-MIT bipolar recordings into 4 s sliding windows with a 2 s causal stride.</span>
                </span>
              </label>

              <label class="qs-step">
                <input type="radio" name="bd-qstep-szcore" value="2" class="sr-only"/>
                <span class="qs-step-num">02</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Predict seizure probabilities</span>
                  <span class="qs-step-desc">Score each window on held-out patients using a trained EEGClassifier or SeizureTransformer.</span>
                </span>
              </label>

              <label class="qs-step">
                <input type="radio" name="bd-qstep-szcore" value="3" class="sr-only"/>
                <span class="qs-step-num">03</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Smooth alarms with causal filters</span>
                  <span class="qs-step-desc">Apply past-only rolling mean and minimum filters so isolated artifacts cannot trigger false alarms.</span>
                </span>
              </label>

              <label class="qs-step">
                <input type="radio" name="bd-qstep-szcore" value="4" class="sr-only"/>
                <span class="qs-step-num">04</span>
                <span class="qs-step-copy">
                  <span class="qs-step-title">Score events with clinical SzCORE</span>
                  <span class="qs-step-desc">Compute overlap F1, seizure sensitivity, and false alarms per 24 hours with szcore_event_scores().</span>
                </span>
              </label>
            </fieldset>
          </div>

          <div class="output-card">
            <div class="output-head">
              <span class="output-label">Pipeline</span>
              <fieldset class="output-tabs" aria-label="Select Python workflow">
                <label class="output-tab">
                  <input type="radio" name="bd-wf" value="moabb" class="sr-only" checked/>
                  <span>train_eegnex.py</span>
                </label>
                <label class="output-tab">
                  <input type="radio" name="bd-wf" value="montage" class="sr-only"/>
                  <span>finetune_labram.py</span>
                </label>
                <label class="output-tab">
                  <input type="radio" name="bd-wf" value="szcore" class="sr-only"/>
                  <span>szcore_seizure.py</span>
                </label>
              </fieldset>
            </div>

            <div class="qs-pane qs-pane-moabb">
              <div class="output-toolbar bd-mono">
                <span class="copy-wrap">
                  <button type="button" class="output-copy-btn" data-copy="from braindecode import EEGClassifier&#10;from braindecode.datasets import MOABBDataset&#10;from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess&#10;&#10;dataset = MOABBDataset(&quot;BNCI2014_001&quot;, subject_ids=[3])&#10;preprocess(dataset, [&#10;    Preprocessor(&quot;pick_types&quot;, eeg=True),&#10;    Preprocessor(&quot;filter&quot;, l_freq=4.0, h_freq=38.0),&#10;])&#10;windows = create_windows_from_events(dataset, preload=True)&#10;clf = EEGClassifier(&quot;EEGNeX&quot;, max_epochs=20, batch_size=64)&#10;clf.fit(windows, y=None)" aria-label="Copy Motor Imagery MOABB workflow">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    <span>Copy Code</span>
                  </button>
                  <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
                </span>
                <span>PyTorch · CUDA / CPU</span>
              </div>
              <div class="output-terminal">
                <pre class="repl-code bd-mono" tabindex="0"><code><span class="cl" data-qstep="1"><span class="tok-com"># 1. Load BCI Competition IV-2a (subject 3) via MOABB</span></span><span class="cl" data-qstep="1"><span class="tok-kw">from</span> braindecode.datasets <span class="tok-kw">import</span> <span class="tok-cls">MOABBDataset</span></span><span class="cl" data-qstep="1">dataset = <span class="tok-cls">MOABBDataset</span>(<span class="tok-str">"BNCI2014_001"</span>, <span class="tok-attr">subject_ids</span>=[<span class="tok-num">3</span>])</span><span class="cl" data-qstep="2"></span><span class="cl" data-qstep="2"><span class="tok-com"># 2. Bandpass-filter EEG channels and slice event windows</span></span><span class="cl" data-qstep="2"><span class="tok-kw">from</span> braindecode.preprocessing <span class="tok-kw">import</span> (</span><span class="cl" data-qstep="2">    <span class="tok-cls">Preprocessor</span>, <span class="tok-fn">create_windows_from_events</span>, <span class="tok-fn">preprocess</span>,</span><span class="cl" data-qstep="2">)</span><span class="cl" data-qstep="2"><span class="tok-fn">preprocess</span>(dataset, [</span><span class="cl" data-qstep="2">    <span class="tok-cls">Preprocessor</span>(<span class="tok-str">"pick_types"</span>, <span class="tok-attr">eeg</span>=<span class="tok-num">True</span>),</span><span class="cl" data-qstep="2">    <span class="tok-cls">Preprocessor</span>(<span class="tok-str">"filter"</span>, <span class="tok-attr">l_freq</span>=<span class="tok-num">4.0</span>, <span class="tok-attr">h_freq</span>=<span class="tok-num">38.0</span>),</span><span class="cl" data-qstep="2">])</span><span class="cl" data-qstep="2">windows = <span class="tok-fn">create_windows_from_events</span>(dataset, <span class="tok-attr">preload</span>=<span class="tok-num">True</span>)</span><span class="cl" data-qstep="3"></span><span class="cl" data-qstep="3"><span class="tok-com"># 3. Pick a published architecture; shapes are derived from windows</span></span><span class="cl" data-qstep="3"><span class="tok-kw">from</span> braindecode <span class="tok-kw">import</span> <span class="tok-cls">EEGClassifier</span></span><span class="cl" data-qstep="3">clf = <span class="tok-cls">EEGClassifier</span>(<span class="tok-str">"EEGNeX"</span>, <span class="tok-attr">max_epochs</span>=<span class="tok-num">20</span>, <span class="tok-attr">batch_size</span>=<span class="tok-num">64</span>)</span><span class="cl" data-qstep="4"></span><span class="cl" data-qstep="4"><span class="tok-com"># 4. Train with scikit-learn compatible fit()</span></span><span class="cl" data-qstep="4">clf.<span class="tok-fn">fit</span>(windows, <span class="tok-attr">y</span>=<span class="tok-num">None</span>)</span></code></pre>
              </div>
              <div class="qs-foot bd-mono">
                <span class="qs-foot-ok">✓ 22 ch · 4 classes · 1000 samples @ 250 Hz derived automatically</span>
                <a class="wf-open-link" href="auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html">Open full tutorial →</a>
              </div>
            </div>

            <div class="qs-pane qs-pane-montage">
              <div class="output-toolbar bd-mono">
                <span class="copy-wrap">
                  <button type="button" class="output-copy-btn" data-copy="import mne&#10;from braindecode import EEGClassifier&#10;from braindecode.models import Labram&#10;&#10;raw = mne.io.read_raw_edf(&quot;patient_eeg.edf&quot;, preload=True)&#10;raw.set_montage(&quot;standard_1020&quot;)&#10;model = Labram.from_pretrained(&#10;    &quot;braindecode/labram-pretrained&quot;,&#10;    chs_info=raw.info[&quot;chs&quot;],&#10;    n_outputs=4,&#10;)&#10;clf = EEGClassifier(model, max_epochs=10, optimizer__lr=1e-4)&#10;clf.fit(windows, y=None)" aria-label="Copy Pretrained Hub LaBraM workflow">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    <span>Copy Code</span>
                  </button>
                  <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
                </span>
                <span>Hugging Face · safetensors</span>
              </div>
              <div class="output-terminal">
                <pre class="repl-code bd-mono" tabindex="0"><code><span class="cl" data-qstep="1"><span class="tok-com"># 1. Read EEG recording with 3D electrode metadata</span></span><span class="cl" data-qstep="1"><span class="tok-kw">import</span> mne</span><span class="cl" data-qstep="1">raw = mne.io.<span class="tok-fn">read_raw_edf</span>(<span class="tok-str">"patient_eeg.edf"</span>, <span class="tok-attr">preload</span>=<span class="tok-num">True</span>)</span><span class="cl" data-qstep="1">raw.<span class="tok-fn">set_montage</span>(<span class="tok-str">"standard_1020"</span>)</span><span class="cl" data-qstep="2"></span><span class="cl" data-qstep="2"><span class="tok-com"># 2. Pull audited LaBraM weights from Hugging Face Hub</span></span><span class="cl" data-qstep="2"><span class="tok-kw">from</span> braindecode.models <span class="tok-kw">import</span> <span class="tok-cls">Labram</span></span><span class="cl" data-qstep="2">model = <span class="tok-cls">Labram</span>.<span class="tok-fn">from_pretrained</span>(</span><span class="cl" data-qstep="2">    <span class="tok-str">"braindecode/labram-pretrained"</span>,</span><span class="cl" data-qstep="3">    <span class="tok-com"># 3. Pass sensor metadata &amp; reset classification head</span></span><span class="cl" data-qstep="3">    <span class="tok-attr">chs_info</span>=raw.info[<span class="tok-str">"chs"</span>],</span><span class="cl" data-qstep="3">    <span class="tok-attr">n_outputs</span>=<span class="tok-num">4</span>,</span><span class="cl" data-qstep="3">)</span><span class="cl" data-qstep="4"></span><span class="cl" data-qstep="4"><span class="tok-com"># 4. Probe frozen embeddings or fine-tune with EEGClassifier</span></span><span class="cl" data-qstep="4"><span class="tok-kw">from</span> braindecode <span class="tok-kw">import</span> <span class="tok-cls">EEGClassifier</span></span><span class="cl" data-qstep="4">clf = <span class="tok-cls">EEGClassifier</span>(model, <span class="tok-attr">max_epochs</span>=<span class="tok-num">10</span>, <span class="tok-attr">optimizer__lr</span>=<span class="tok-num">1e-4</span>)</span><span class="cl" data-qstep="4">clf.<span class="tok-fn">fit</span>(windows, <span class="tok-attr">y</span>=<span class="tok-num">None</span>)</span></code></pre>
              </div>
              <div class="qs-foot bd-mono">
                <span class="qs-foot-ok">✓ Audited safetensors weights + automatic task-head reset</span>
                <a class="wf-open-link" href="auto_examples/model_building/plot_load_pretrained_models.html">Open Hub tutorial →</a>
              </div>
            </div>

            <div class="qs-pane qs-pane-szcore">
              <div class="output-toolbar bd-mono">
                <span class="copy-wrap">
                  <button type="button" class="output-copy-btn" data-copy="import numpy as np, pandas as pd&#10;from braindecode.preprocessing import create_fixed_length_windows&#10;from braindecode.training import szcore_event_scores&#10;&#10;windows = create_fixed_length_windows(test_ds, window_size_samples=1024, window_stride_samples=512, preload=True)&#10;prob = clf.predict_proba(windows)[:, 1]&#10;starts = windows.get_metadata()[&quot;i_start_in_trial&quot;].to_numpy() / 256.0&#10;smooth = pd.Series(prob).rolling(5).mean().rolling(3).min().to_numpy()&#10;alarms = starts[smooth &gt; 0.9]&#10;events = np.c_[alarms + 2.0, alarms + 4.0]&#10;scores = szcore_event_scores(ref_seizures, events, duration_s)" aria-label="Copy Clinical Seizure Event Scoring workflow">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    <span>Copy Code</span>
                  </button>
                  <span class="bd-copy-status sr-only" role="status" aria-live="polite"></span>
                </span>
                <span>Clinical · SzCORE</span>
              </div>
              <div class="output-terminal">
                <pre class="repl-code bd-mono" tabindex="0"><code><span class="cl" data-qstep="1"><span class="tok-com"># 1. Cut held-out CHB-MIT recording into 4 s windows (2 s stride @ 256 Hz)</span></span><span class="cl" data-qstep="1"><span class="tok-kw">import</span> numpy <span class="tok-kw">as</span> np, pandas <span class="tok-kw">as</span> pd</span><span class="cl" data-qstep="1"><span class="tok-kw">from</span> braindecode.preprocessing <span class="tok-kw">import</span> <span class="tok-fn">create_fixed_length_windows</span></span><span class="cl" data-qstep="1">windows = <span class="tok-fn">create_fixed_length_windows</span>(</span><span class="cl" data-qstep="1">    test_ds, <span class="tok-attr">window_size_samples</span>=<span class="tok-num">1024</span>, <span class="tok-attr">window_stride_samples</span>=<span class="tok-num">512</span>, <span class="tok-attr">preload</span>=<span class="tok-num">True</span>,</span><span class="cl" data-qstep="1">)</span><span class="cl" data-qstep="2"></span><span class="cl" data-qstep="2"><span class="tok-com"># 2. Predict seizure probability for each window</span></span><span class="cl" data-qstep="2">prob = clf.<span class="tok-fn">predict_proba</span>(windows)[:, <span class="tok-num">1</span>]</span><span class="cl" data-qstep="2">starts = windows.<span class="tok-fn">get_metadata</span>()[<span class="tok-str">"i_start_in_trial"</span>].<span class="tok-fn">to_numpy</span>() / <span class="tok-num">256.0</span></span><span class="cl" data-qstep="3"></span><span class="cl" data-qstep="3"><span class="tok-com"># 3. Smooth causally over past windows and extract alarm intervals</span></span><span class="cl" data-qstep="3">smooth = pd.<span class="tok-cls">Series</span>(prob).<span class="tok-fn">rolling</span>(<span class="tok-num">5</span>).<span class="tok-fn">mean</span>().<span class="tok-fn">rolling</span>(<span class="tok-num">3</span>).<span class="tok-fn">min</span>().<span class="tok-fn">to_numpy</span>()</span><span class="cl" data-qstep="3">alarms = starts[smooth &gt; <span class="tok-num">0.9</span>]</span><span class="cl" data-qstep="3">events = np.c_[alarms + <span class="tok-num">2.0</span>, alarms + <span class="tok-num">4.0</span>]</span><span class="cl" data-qstep="4"></span><span class="cl" data-qstep="4"><span class="tok-com"># 4. Score overlap sensitivity, F1 &amp; false alarms/24h under SzCORE</span></span><span class="cl" data-qstep="4"><span class="tok-kw">from</span> braindecode.training <span class="tok-kw">import</span> <span class="tok-fn">szcore_event_scores</span></span><span class="cl" data-qstep="4">scores = <span class="tok-fn">szcore_event_scores</span>(ref_seizures, events, duration_s)</span></code></pre>
              </div>
              <div class="qs-foot bd-mono">
                <span class="qs-foot-ok">✓ Overlap F1, sensitivity &amp; false-alarm rate per 24h</span>
                <a class="wf-open-link" href="auto_examples/applied_examples/plot_seizure_alarm_chbmit.html">Open clinical tutorial →</a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section zoo-section" id="model-zoo">
      <div class="container">
        <div class="zoo-head-center">
          <span class="section-eyebrow">MODEL ZOO · <span data-count-to="103">103</span> ARCHITECTURES</span>
          <h2 class="section-title is-center">Explore <span class="zoo-count-num" data-count-to="103">103</span> published architectures</h2>
          <p class="zoo-sub">Every model is reproduced from its original paper and ships with a consistent constructor. The shared <code class="bd-mono">EEGModuleMixin</code> takes any of <code class="bd-mono">n_chans</code>, <code class="bd-mono">n_outputs</code>, <code class="bd-mono">n_times</code>, <code class="bd-mono">chs_info</code>, <code class="bd-mono">input_window_seconds</code>, or <code class="bd-mono">sfreq</code>; the missing ones are derived for you. Hover any pill or card to preview its Python API.</p>
        </div>

.. raw:: html
    :file: generated/zoo_browser.html

.. raw:: html

      </div>
    </section>

    <section class="section tut-section">
      <div class="container">
        <div class="tut-head">
          <div>
            <span class="section-eyebrow">EXECUTED GALLERY</span>
            <h2 class="section-title is-left">Runnable tutorials from executed notebooks</h2>
          </div>
          <a class="btn" href="auto_examples/index.html">Browse all tutorials <span aria-hidden="true">→</span></a>
        </div>

        <div class="tut-grid">
          <a class="tut-card" href="auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html">
            <img class="tut-thumb dark-light" src="_images/sphx_glr_plot_bcic_iv_2a_moabb_trial_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Motor Imagery · EEG</span>
            <h3 class="tut-title">Trialwise decoding on BCI Competition IV-2a</h3>
          </a>
          <a class="tut-card" href="auto_examples/model_building/plot_load_pretrained_models.html">
            <img class="tut-thumb dark-light" src="_images/sphx_glr_plot_load_pretrained_models_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Hugging Face Hub</span>
            <h3 class="tut-title">Load, adapt, and audit pretrained foundation models</h3>
          </a>
          <a class="tut-card" href="auto_examples/model_building/plot_channel_interpolation.html">
            <img class="tut-thumb dark-light" src="_images/sphx_glr_plot_channel_interpolation_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Montage · ChannelLayer</span>
            <h3 class="tut-title">Exact, spherical-spline, and source montage interpolation</h3>
          </a>
          <a class="tut-card" href="auto_examples/advanced_training/plot_sae_reve.html">
            <img class="tut-thumb dark-light" src="_images/sphx_glr_plot_sae_reve_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Mechanistic XAI · REVE</span>
            <h3 class="tut-title">Causal sparse-autoencoder feature interventions on REVE</h3>
          </a>
          <a class="tut-card" href="auto_examples/applied_examples/plot_sleep_staging_chambon2018.html">
            <img class="tut-thumb dark-light" src="_images/sphx_glr_plot_sleep_staging_chambon2018_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Sleep PSG</span>
            <h3 class="tut-title">Clinical polysomnography staging on Sleep Physionet</h3>
          </a>
          <a class="tut-card" href="auto_examples/advanced_training/plot_interpretability.html">
            <img class="tut-thumb dark-light" src="_images/sphx_glr_plot_interpretability_thumb.png" alt="" width="400" height="280" loading="lazy" decoding="async"/>
            <span class="tut-tag bd-mono">Attribution &amp; XAI</span>
            <h3 class="tut-title">Amplitude-spectrum gradients and randomization checks</h3>
          </a>
        </div>
      </div>
    </section>

    <section class="section eco-section" id="ecosystem">
      <div class="container">
        <div class="section-head-stack is-center">
          <span class="section-eyebrow">OPEN ECOSYSTEM · 2.1M+ INSTALLS</span>
          <h2 class="section-title">Built and shared with neighbors</h2>
          <p class="section-sub">Open neural decoding does not live alone. Braindecode plugs into a wider open-source neuroscience stack, ships sibling projects that round out the data story, and is built upon across MNE-Python, EEGDash, MOABB, NeuralBench, NeuroAI, Meta Brain &amp; AI, SPD Learn, EEGPrep, and skorch.</p>
        </div>

        <div class="eco-grid is-nine">
          <a class="eco-card" href="https://mne.tools/" target="_blank" rel="noopener">
            <div class="eco-top">
              <img class="eco-logo only-light" src="_static/ecosystem_logos/mne-logo.svg" alt="MNE-Python" width="114" height="36" loading="lazy" decoding="async"/>
              <img class="eco-logo only-dark" src="_static/ecosystem_logos/mne-logo-dark.svg" alt="MNE-Python" width="114" height="36" loading="lazy" decoding="async"/>
              <span class="eco-badge bd-mono">Signal foundation</span>
            </div>
            <h3 class="eco-name">MNE-Python</h3>
            <p class="eco-desc">Direct ingestion of <code>mne.io.Raw</code>, <code>mne.Epochs</code>, BIDS metadata, and 3D sensor montages.</p>
          </a>

          <a class="eco-card" href="https://eegdash.org/" target="_blank" rel="noopener">
            <div class="eco-top">
              <img class="eco-logo only-light" src="_static/ecosystem_logos/eegdash-long.svg" alt="EEGDash" width="144" height="32" loading="lazy" decoding="async"/>
              <img class="eco-logo only-dark" src="_static/ecosystem_logos/eegdash-long-dark.svg" alt="EEGDash" width="144" height="32" loading="lazy" decoding="async"/>
              <span class="eco-badge bd-mono">954 BIDS corpora</span>
            </div>
            <h3 class="eco-name">EEGDash</h3>
            <p class="eco-desc">954 BIDS-curated EEG, MEG, iEEG, fNIRS, and EMG datasets across 48,000+ subjects and 138,000+ hours returning PyTorch-ready Braindecode datasets.</p>
          </a>

          <a class="eco-card" href="https://moabb.neurotechx.com/" target="_blank" rel="noopener">
            <div class="eco-top">
              <img class="eco-logo only-light" src="_static/ecosystem_logos/moabb-light.svg" alt="MOABB" width="132" height="32" loading="lazy" decoding="async"/>
              <img class="eco-logo only-dark" src="_static/ecosystem_logos/moabb-dark.svg" alt="MOABB" width="132" height="32" loading="lazy" decoding="async"/>
              <span class="eco-badge bd-mono">158 benchmarks</span>
            </div>
            <h3 class="eco-name">MOABB</h3>
            <p class="eco-desc">Leak-free within-session, cross-session, and cross-subject BCI benchmarks wrapping 158 open datasets via <code>MOABBDataset</code>.</p>
          </a>

          <a class="eco-card" href="https://facebookresearch.github.io/neuroai/neuralbench/" target="_blank" rel="noopener">
            <div class="eco-top">
              <img class="eco-logo only-light" src="_static/ecosystem_logos/neuralbench-on-light.svg" alt="NeuralBench" width="154" height="30" loading="lazy" decoding="async"/>
              <img class="eco-logo only-dark" src="_static/ecosystem_logos/neuralbench-on-dark.svg" alt="NeuralBench" width="154" height="30" loading="lazy" decoding="async"/>
              <span class="eco-badge bd-mono">FM benchmarks</span>
            </div>
            <h3 class="eco-name">NeuralBench</h3>
            <p class="eco-desc">Standardized evaluation across EEG, MEG, fNIRS, and sEMG tasks built directly on Braindecode models and Hub checkpoints.</p>
          </a>

          <a class="eco-card" href="https://github.com/facebookresearch/neuroai" target="_blank" rel="noopener">
            <div class="eco-top">
              <img class="eco-logo only-light" src="_static/ecosystem_logos/neuroai-on-light.svg" alt="NeuroAI" width="138" height="34" loading="lazy" decoding="async"/>
              <img class="eco-logo only-dark" src="_static/ecosystem_logos/neuroai-on-dark.svg" alt="NeuroAI" width="138" height="34" loading="lazy" decoding="async"/>
              <span class="eco-badge bd-mono">Neuro-AI suite</span>
            </div>
            <h3 class="eco-name">NeuroAI</h3>
            <p class="eco-desc">Modular Python suite (<code>NeuralSet</code>, <code>NeuralFetch</code>, <code>NeuralTrain</code>, <code>NeuralBench</code>) for large-scale neurophysiology training.</p>
          </a>

          <a class="eco-card" href="https://neural-interfaces26.github.io/" target="_blank" rel="noopener">
            <div class="eco-top">
              <img class="eco-logo only-light" src="_static/ecosystem_logos/meta-brainai-light.png" alt="Meta Brain &amp; AI" width="142" height="30" loading="lazy" decoding="async"/>
              <img class="eco-logo only-dark" src="_static/ecosystem_logos/meta-brainai-dark.png" alt="Meta Brain &amp; AI" width="142" height="30" loading="lazy" decoding="async"/>
              <span class="eco-badge bd-mono">Challenge partner</span>
            </div>
            <h3 class="eco-name">Meta Brain &amp; AI</h3>
            <p class="eco-desc">Co-leading NeurIPS 2026 EEG/EMG Foundation Challenge tracks and open-source non-invasive brain decoding models.</p>
          </a>

          <a class="eco-card" href="https://spdlearn.org/" target="_blank" rel="noopener">
            <div class="eco-top">
              <span class="eco-brand-inline">
                <img class="eco-logo only-light" src="_static/ecosystem_logos/spd-learn-mark.png" alt="" width="38" height="40" loading="lazy" decoding="async"/>
                <img class="eco-logo only-dark" src="_static/ecosystem_logos/spd-learn-mark-dark.png" alt="" width="38" height="40" loading="lazy" decoding="async"/>
                <span class="eco-brand-word">SPD Learn</span>
              </span>
              <span class="eco-badge bd-mono">Riemannian DL</span>
            </div>
            <h3 class="eco-name">SPD Learn</h3>
            <p class="eco-desc">Sibling PyTorch library for geometric deep learning on Symmetric Positive Definite covariance matrices (<code>SPDNet</code>, <code>TSMNet</code>, <code>EEGSPDNet</code>).</p>
          </a>

          <a class="eco-card" href="https://sccn.github.io/eegprep/" target="_blank" rel="noopener">
            <div class="eco-top">
              <img class="eco-logo only-light" src="_static/ecosystem_logos/eegprep-light-transparent.png" alt="EEGPrep" width="56" height="46" loading="lazy" decoding="async"/>
              <img class="eco-logo only-dark" src="_static/ecosystem_logos/eegprep-dark-transparent.png" alt="EEGPrep" width="56" height="46" loading="lazy" decoding="async"/>
              <span class="eco-badge bd-mono">Artifact cleaning</span>
            </div>
            <h3 class="eco-name">EEGPrep</h3>
            <p class="eco-desc">Automated SCCN/EEGLAB artifact subspace reconstruction (ASR), bad-channel rejection, and spherical interpolation in <code>braindecode.preprocessing</code>.</p>
          </a>

          <a class="eco-card" href="https://skorch.readthedocs.io/" target="_blank" rel="noopener">
            <div class="eco-top">
              <img class="eco-logo only-light" src="_static/ecosystem_logos/skorch-logo.svg" alt="skorch" width="118" height="28" loading="lazy" decoding="async"/>
              <img class="eco-logo only-dark" src="_static/ecosystem_logos/skorch-dark.svg" alt="skorch" width="118" height="28" loading="lazy" decoding="async"/>
              <span class="eco-badge bd-mono">Estimator API</span>
            </div>
            <h3 class="eco-name">skorch</h3>
            <p class="eco-desc">Powers <code>EEGClassifier</code> and <code>EEGRegressor</code> with scikit-learn <code>fit()</code>, <code>predict_proba()</code>, and cross-validation.</p>
          </a>
        </div>

        <p class="eco-foot">
          Building on Braindecode? <a href="https://github.com/braindecode/braindecode/network/dependents" target="_blank" rel="noopener">Explore dependent projects on GitHub →</a>
        </p>
      </div>
    </section>

    <section class="closing-band">
      <div class="container closing-inner">
        <img class="closing-wordmark dark-light" src="_static/braindecode_long_dark.svg" alt="Braindecode" width="352" height="57" loading="lazy" decoding="async"/>
        <h2 class="section-title is-center is-white">Start decoding brain signals today</h2>
        <p class="closing-sub">Install in one command, pull a pretrained foundation backbone or a MOABB / NeuralBench benchmark, and run your first experiment in minutes.</p>
        <div class="hero-cta-row">
          <a class="btn btn-solid-light" href="install/install.html">Get started <span aria-hidden="true">→</span></a>
          <a class="btn btn-ghost-light" href="https://huggingface.co/braindecode" target="_blank" rel="noopener">Hugging Face Hub</a>
          <a class="btn btn-ghost-light" href="https://github.com/braindecode/braindecode" target="_blank" rel="noopener">Star on GitHub</a>
        </div>
      </div>
    </section>

    <footer class="ruled-footer">
      <div class="container ruled-footer-grid">
        <div class="rf-brand">
          <div class="rf-brand-top">
            <div class="rf-wordmark-wrap">
              <img class="footer-logo only-light" src="_static/braindecode_long.svg" alt="Braindecode" width="272" height="44" loading="lazy" decoding="async"/>
              <img class="footer-logo only-dark" src="_static/braindecode_long_dark.svg" alt="Braindecode" width="272" height="44" loading="lazy" decoding="async"/>
            </div>
            <p class="rf-tagline">Open-source toolbox for decoding raw EEG, iEEG, MEG, fNIRS, and sEMG with deep learning. Built on PyTorch, MNE-Python, and skorch.</p>
            <div class="rf-quick">
              <a href="whats_new.html">Release notes</a>
              <a href="agent_docs.html">Agent Markdown &amp; llms.txt</a>
              <a href="https://github.com/braindecode/braindecode" target="_blank" rel="noopener">Star on GitHub →</a>
            </div>
          </div>

          <div class="rf-cite-card">
            <span class="rf-cite-kicker bd-mono">CITE BRAINDECODE · SOFTWARE &amp; PAPER</span>
            <a class="rf-doi-badge bd-mono" href="https://doi.org/10.5281/zenodo.8214376" target="_blank" rel="noopener">
              <span class="rf-doi-tag">Zenodo DOI</span>
              <b>10.5281/zenodo.8214376</b>
            </a>
            <a class="rf-doi-sub bd-mono" href="https://doi.org/10.1002/hbm.23730" target="_blank" rel="noopener">HBM 2017 · 10.1002/hbm.23730</a>
            <a class="rf-cite-guide bd-mono" href="cite.html">How to cite Braindecode →</a>
          </div>
        </div>

        <div class="rf-right">
          <div class="rf-cells">
            <div class="rf-cell">
              <h3>Get started</h3>
              <ul>
                <li><a href="install/install.html">Installation</a></li>
                <li><a href="auto_examples/index.html">Tutorials</a></li>
                <li><a href="models/channel_strategies.html">Channel strategies</a></li>
                <li><a href="auto_examples/model_building/plot_bcic_iv_2a_moabb_cropped.html#cropped-inference">Cropped inference</a></li>
                <li><a href="api.html">API reference</a></li>
              </ul>
            </div>

            <div class="rf-cell">
              <h3>Models &amp; Hub</h3>
              <ul>
                <li><a href="models/models_categorization.html">Architecture taxonomy</a></li>
                <li><a href="models/models_table.html">Models summary table</a></li>
                <li><a href="models/categorization/lbm.html">Foundation models</a></li>
                <li><a href="https://huggingface.co/braindecode" target="_blank" rel="noopener">Hugging Face Hub</a></li>
                <li><a href="https://huggingface.co/spaces/braindecode/OpenEEGBench" target="_blank" rel="noopener">OpenEEG-Bench</a></li>
              </ul>
            </div>

            <div class="rf-cell">
              <h3>Developers</h3>
              <ul>
                <li><a href="https://github.com/braindecode/braindecode" target="_blank" rel="noopener">GitHub repository</a></li>
                <li><a href="https://github.com/braindecode/braindecode/discussions" target="_blank" rel="noopener">Discussions</a></li>
                <li><a href="llms.txt">llms.txt index</a></li>
                <li><a href="help.html">Help &amp; support</a></li>
              </ul>
            </div>

            <div class="rf-cell">
              <h3>Ecosystem</h3>
              <ul>
                <li><a href="https://mne.tools/" target="_blank" rel="noopener">MNE-Python</a></li>
                <li><a href="https://eegdash.org/" target="_blank" rel="noopener">EEGDash</a></li>
                <li><a href="https://moabb.neurotechx.com/" target="_blank" rel="noopener">MOABB</a></li>
                <li><a href="https://facebookresearch.github.io/neuroai/neuralbench/" target="_blank" rel="noopener">NeuralBench</a></li>
                <li><a href="https://github.com/facebookresearch/neuroai" target="_blank" rel="noopener">NeuroAI</a></li>
                <li><a href="https://spdlearn.org/" target="_blank" rel="noopener">SPD Learn</a></li>
                <li><a href="https://sccn.github.io/eegprep/" target="_blank" rel="noopener">EEGPrep</a></li>
                <li><a href="https://skorch.readthedocs.io/" target="_blank" rel="noopener">skorch</a></li>
              </ul>
            </div>
          </div>

          <div class="rf-bottom">
            <div class="rf-actions">
              <button type="button" class="btn btn-sm btn-primary" data-bd-cite-copy="zenodo-bibtex" aria-label="Copy Zenodo BibTeX citation for Braindecode software">Copy Zenodo BibTeX</button>
              <button type="button" class="btn btn-sm" data-bd-cite-copy="both-bibtex" aria-label="Copy both Zenodo software and HBM 2017 paper BibTeX citations">Copy Paper + Software BibTeX</button>
              <a class="btn btn-sm" href="_static/braindecode.bib" download="braindecode.bib">Download .bib</a>
            </div>
            <div class="rf-meta-row">
              <span class="rf-copy">© 2017–2026 The Braindecode Developers · BSD-3-Clause License</span>
              <a class="rf-cite-inline bd-mono" href="cite.html">How to cite Braindecode →</a>
            </div>
          </div>
        </div>
      </div>
    </footer>

    </div>

.. toctree::
    :hidden:
    :caption: Get Started

    Install <install/install>
    Tutorials <auto_examples/index>
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
    Help <help>
