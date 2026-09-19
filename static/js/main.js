/**
 * ChestVision AI - Client Diagnostic Controller
 * Manages drag-and-drop, sample X-rays, multi-step scan animations,
 * asynchronous inference requests, and diagnostic report generation.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const selectFileBtn = document.getElementById('selectFileBtn');
  const previewContainer = document.getElementById('previewContainer');
  const previewImage = document.getElementById('previewImage');
  const scanLine = document.getElementById('scanLine');
  const changeImageBtn = document.getElementById('changeImageBtn');
  const runAnalyzeBtn = document.getElementById('runAnalyzeBtn');

  // Results State Elements
  const emptyState = document.getElementById('emptyState');
  const loadingState = document.getElementById('loadingState');
  const resultsContent = document.getElementById('resultsContent');
  const loadingStepMsg = document.getElementById('loadingStepMsg');

  // Results Data Elements
  const diagnosisBanner = document.getElementById('diagnosisBanner');
  const diagnosisName = document.getElementById('diagnosisName');
  const confidenceNumber = document.getElementById('confidenceNumber');
  const probabilityBarsList = document.getElementById('probabilityBarsList');
  const clinicalSummary = document.getElementById('clinicalSummary');
  const clinicalBullets = document.getElementById('clinicalBullets');
  const latencyBadge = document.getElementById('latencyBadge');
  const deviceBadge = document.getElementById('deviceBadge');
  const printReportBtn = document.getElementById('printReportBtn');
  const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');

  // Sample cards
  const sampleCards = document.querySelectorAll('.sample-card');

  let currentFile = null;
  let currentSampleName = null;
  let scanAnimationTimer = null;

  // -------------------------------------------------------------------------
  // Drag and Drop Listeners
  // -------------------------------------------------------------------------
  selectFileBtn.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('click', (e) => {
    if (e.target !== selectFileBtn) fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-over');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  function handleFileSelection(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (PNG, JPG, JPEG).');
      return;
    }

    currentFile = file;
    currentSampleName = null;

    const reader = new FileReader();
    reader.onload = (e) => {
      displayPreview(e.target.result);
    };
    reader.readAsDataURL(file);
  }

  function displayPreview(imageSrc) {
    previewImage.src = imageSrc;
    dropzone.style.display = 'none';
    previewContainer.style.display = 'flex';
    resetResultsToEmpty();
  }

  changeImageBtn.addEventListener('click', () => {
    currentFile = null;
    currentSampleName = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    dropzone.style.display = 'flex';
    stopScanAnimation();
    resetResultsToEmpty();
  });

  // -------------------------------------------------------------------------
  // Sample X-Rays Fast Select
  // -------------------------------------------------------------------------
  sampleCards.forEach(card => {
    card.addEventListener('click', () => {
      const sampleFile = card.getAttribute('data-sample');
      const sampleThumb = card.querySelector('img').src;
      currentFile = null;
      currentSampleName = sampleFile;

      displayPreview(sampleThumb);
      // Auto-trigger analysis for seamless instant testing
      executeInference();
    });
  });

  runAnalyzeBtn.addEventListener('click', () => {
    executeInference();
  });

  // -------------------------------------------------------------------------
  // Inference Execution & Scan Animation
  // -------------------------------------------------------------------------
  function startScanAnimation() {
    scanLine.style.display = 'block';
    emptyState.style.display = 'none';
    resultsContent.style.display = 'none';
    loadingState.style.display = 'flex';

    const steps = [
      'Pre-processing chest radiograph to 224x224 tensor...',
      'Extracting spatial lung patterns via 5-Stage CNN...',
      'Processing visual tokens with Transformer Attention...',
      'Synthesizing probability distributions...'
    ];
    let stepIndex = 0;
    loadingStepMsg.textContent = steps[0];

    scanAnimationTimer = setInterval(() => {
      stepIndex = (stepIndex + 1) % steps.length;
      loadingStepMsg.textContent = steps[stepIndex];
    }, 450);
  }

  function stopScanAnimation() {
    scanLine.style.display = 'none';
    if (scanAnimationTimer) {
      clearInterval(scanAnimationTimer);
      scanAnimationTimer = null;
    }
  }

  function resetResultsToEmpty() {
    stopScanAnimation();
    loadingState.style.display = 'none';
    resultsContent.style.display = 'none';
    emptyState.style.display = 'flex';
  }

  async function executeInference() {
    if (!currentFile && !currentSampleName) {
      alert('Please upload a chest X-ray image or pick a sample scan first.');
      return;
    }

    startScanAnimation();
    runAnalyzeBtn.disabled = true;

    try {
      let response;
      if (currentFile) {
        const formData = new FormData();
        formData.append('image', currentFile);
        response = await fetch('/predict', {
          method: 'POST',
          body: formData
        });
      } else if (currentSampleName) {
        response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sample: currentSampleName })
        });
      }

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Server returned an error during analysis.');
      }

      // Small deliberate delay so user experiences the scanning state
      setTimeout(() => {
        renderResults(data);
      }, 500);

    } catch (err) {
      stopScanAnimation();
      loadingState.style.display = 'none';
      emptyState.style.display = 'flex';
      alert('Diagnosis error: ' + err.message);
    } finally {
      runAnalyzeBtn.disabled = false;
    }
  }

  // -------------------------------------------------------------------------
  // Render Diagnosis & Probability Graphs
  // -------------------------------------------------------------------------
  function renderResults(data) {
    stopScanAnimation();
    loadingState.style.display = 'none';
    emptyState.style.display = 'none';
    resultsContent.style.display = 'flex';

    // 1. Diagnosis Banner
    diagnosisBanner.className = 'diagnosis-banner ' + (data.metadata.badge_class || '');
    diagnosisName.textContent = data.prediction;

    // Animate confidence number
    animateCounter(confidenceNumber, 0, data.confidence, 700);

    // 2. Probability Distribution Bars
    probabilityBarsList.innerHTML = '';
    data.probabilities.forEach(item => {
      const probItem = document.createElement('div');
      probItem.className = 'prob-item';

      const safeClass = item.class_name.replace(/\s+/g, '-');

      probItem.innerHTML = `
        <div class="prob-header">
          <span>${item.class_name}</span>
          <span style="font-family: var(--font-mono);">${item.percentage}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill fill-${safeClass}" style="width: 0%;"></div>
        </div>
      `;
      probabilityBarsList.appendChild(probItem);

      // Staggered fill animation
      setTimeout(() => {
        const fillBar = probItem.querySelector('.prob-fill');
        if (fillBar) fillBar.style.width = item.percentage + '%';
      }, 50);
    });

    // 3. Clinical Overview & Findings
    clinicalSummary.textContent = data.metadata.summary || '';
    clinicalBullets.innerHTML = '';

    const allNotes = [
      ...(data.metadata.findings || []),
      ...(data.metadata.recommendations || [])
    ];

    allNotes.forEach(note => {
      const li = document.createElement('li');
      li.textContent = note;
      clinicalBullets.appendChild(li);
    });

    // 4. Metadata footer
    latencyBadge.textContent = `Inference: ${data.latency_ms} ms`;
    deviceBadge.textContent = `Engine: PyTorch (${data.device})`;
  }

  function animateCounter(element, start, end, duration) {
    const range = end - start;
    const minTimer = 20;
    let stepTime = Math.abs(Math.floor(duration / range));
    stepTime = Math.max(stepTime, minTimer);

    const startTime = new Date().getTime();
    const endTime = startTime + duration;

    function run() {
      const now = new Date().getTime();
      const remaining = Math.max((endTime - now) / duration, 0);
      const value = Math.round(end - (remaining * range));
      element.textContent = value + '%';
      if (value !== end) {
        requestAnimationFrame(run);
      }
    }
    requestAnimationFrame(run);
  }

  // Action Buttons
  printReportBtn.addEventListener('click', () => {
    window.print();
  });

  analyzeAnotherBtn.addEventListener('click', () => {
    changeImageBtn.click();
  });
});
