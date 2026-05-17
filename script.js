const MAX_FILE_SIZE = 10 * 1024 * 1024;
const SUPPORTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_SOURCE_PIXELS = 12_000_000;
const MAX_OUTPUT_PIXELS = 22_000_000;
const DEFAULT_COMPARE_VALUE = 50;
const TOAST_DURATION = 3200;
const ORT_WEB_VERSION = "1.22.0";
const ORT_WEBGPU_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_WEB_VERSION}/dist/ort.webgpu.min.js`;
const AI_TILE_OVERLAP = 16;
const AI_MODEL_SCALE = 4;

const AI_MODEL_CONFIG = {
  fast: {
    label: "Fast",
    profile: "general",
    url: "./models/realesrgan-general-x4.onnx",
    tileSize: 160,
  },
  balanced: {
    label: "Balanced",
    profile: "portrait",
    url: "./models/realesrgan-portrait-x4.onnx",
    tileSize: 192,
  },
  quality: {
    label: "High Quality",
    profile: "anime",
    url: "./models/realesrgan-anime-x4.onnx",
    tileSize: 224,
  },
};

const MODE_CONFIG = {
  fast: {
    denoiseBias: 0.1,
    edgeBoost: 0.32,
    detailBoost: 0.12,
    textBoost: 0.1,
    saturationBoost: 0.03,
    estimateFactor: 1,
  },
  balanced: {
    denoiseBias: 0.16,
    edgeBoost: 0.5,
    detailBoost: 0.2,
    textBoost: 0.18,
    saturationBoost: 0.06,
    estimateFactor: 1.2,
  },
  quality: {
    denoiseBias: 0.22,
    edgeBoost: 0.68,
    detailBoost: 0.28,
    textBoost: 0.28,
    saturationBoost: 0.08,
    estimateFactor: 1.45,
  },
};

const elements = {
  fileInput: document.getElementById("fileInput"),
  topbarUploadBtn: document.getElementById("topbarUploadBtn"),
  uploadBtn: document.getElementById("uploadBtn"),
  upscaleAllBtn: document.getElementById("upscaleAllBtn"),
  cancelBtn: document.getElementById("cancelBtn"),
  zipBtn: document.getElementById("zipBtn"),
  clearCompletedBtn: document.getElementById("clearCompletedBtn"),
  resetAllBtn: document.getElementById("resetAllBtn"),
  dropzone: document.getElementById("dropzone"),
  queueGrid: document.getElementById("queueGrid"),
  emptyQueue: document.getElementById("emptyQueue"),
  statusMessage: document.getElementById("statusMessage"),
  queueCount: document.getElementById("queueCount"),
  completedCount: document.getElementById("completedCount"),
  failedCount: document.getElementById("failedCount"),
  pendingCount: document.getElementById("pendingCount"),
  processingCount: document.getElementById("processingCount"),
  doneCount: document.getElementById("doneCount"),
  batchFailedCount: document.getElementById("batchFailedCount"),
  batchProgressFill: document.getElementById("batchProgressFill"),
  batchProgressText: document.getElementById("batchProgressText"),
  batchProgressLabel: document.getElementById("batchProgressLabel"),
  modeSelect: document.getElementById("modeSelect"),
  scaleSelect: document.getElementById("scaleSelect"),
  formatSelect: document.getElementById("formatSelect"),
  qualitySlider: document.getElementById("qualitySlider"),
  qualityValue: document.getElementById("qualityValue"),
  sharpnessSlider: document.getElementById("sharpnessSlider"),
  sharpnessValue: document.getElementById("sharpnessValue"),
  aiStatus: document.getElementById("aiStatus"),
  safeModeBadge: document.getElementById("safeModeBadge"),
  selectedTitle: document.getElementById("selectedTitle"),
  selectedMeta: document.getElementById("selectedMeta"),
  originalPreview: document.getElementById("originalPreview"),
  originalEmpty: document.getElementById("originalEmpty"),
  resultPreview: document.getElementById("resultPreview"),
  resultEmpty: document.getElementById("resultEmpty"),
  originalMeta: document.getElementById("originalMeta"),
  resultMeta: document.getElementById("resultMeta"),
  downloadSelectedBtn: document.getElementById("downloadSelectedBtn"),
  compareBase: document.getElementById("compareBase"),
  compareEnhanced: document.getElementById("compareEnhanced"),
  compareOverlay: document.getElementById("compareOverlay"),
  compareLine: document.getElementById("compareLine"),
  compareSlider: document.getElementById("compareSlider"),
  compareEmpty: document.getElementById("compareEmpty"),
  loader: document.getElementById("loader"),
  progressValue: document.getElementById("progressValue"),
  progressStage: document.getElementById("progressStage"),
  toastStack: document.getElementById("toastStack"),
};

const state = {
  queue: [],
  selectedId: "",
  compareValue: DEFAULT_COMPARE_VALUE,
  isBatchProcessing: false,
  cancelRequested: false,
  activeJobId: 0,
  aiWorker: null,
  pixelWorker: null,
  pendingAiResolvers: new Map(),
  pendingPixelResolvers: new Map(),
};

initialize();

function initialize() {
  bindEvents();
  ensureAiWorker();
  ensurePixelWorker();
  syncControlLabels();
  updateSafeModeBadge();
  updateComparison(DEFAULT_COMPARE_VALUE);
  render();
}

function bindEvents() {
  elements.fileInput.addEventListener("change", handleFileInputChange);
  elements.topbarUploadBtn.addEventListener("click", () => elements.fileInput.click());
  elements.uploadBtn.addEventListener("click", () => elements.fileInput.click());
  elements.upscaleAllBtn.addEventListener("click", processBatch);
  elements.cancelBtn.addEventListener("click", cancelBatch);
  elements.zipBtn.addEventListener("click", downloadAllAsZip);
  elements.clearCompletedBtn.addEventListener("click", clearCompletedItems);
  elements.resetAllBtn.addEventListener("click", resetQueue);
  elements.downloadSelectedBtn.addEventListener("click", downloadSelectedResult);
  elements.compareSlider.addEventListener("input", (event) => {
    updateComparison(Number(event.target.value));
  });
  elements.qualitySlider.addEventListener("input", syncControlLabels);
  elements.sharpnessSlider.addEventListener("input", syncControlLabels);
  elements.modeSelect.addEventListener("change", () => {
    updateStatus(`Mode set to ${capitalize(elements.modeSelect.value)}.`, "success");
  });

  bindDropzone();
  bindClipboardPaste();
}

function bindDropzone() {
  ["dragenter", "dragover"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "dragend", "drop"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.remove("is-dragover");
    });
  });

  elements.dropzone.addEventListener("drop", (event) => {
    const files = Array.from(event.dataTransfer?.files || []);
    if (files.length > 0) {
      addFilesToQueue(files);
    }
  });
}

function bindClipboardPaste() {
  window.addEventListener("paste", (event) => {
    const files = getFilesFromClipboard(event.clipboardData);
    if (files.length === 0) {
      return;
    }

    event.preventDefault();
    addFilesToQueue(files);
    toast("Images Added", `${files.length} clipboard image${files.length > 1 ? "s" : ""} added.`, "success");
  });
}

function handleFileInputChange(event) {
  const files = Array.from(event.target.files || []);
  if (files.length > 0) {
    addFilesToQueue(files);
  }
  elements.fileInput.value = "";
}

async function addFilesToQueue(files) {
  if (state.isBatchProcessing) {
    toast("Batch Running", "Cancel the current batch before adding more files.", "error");
    return;
  }

  let addedCount = 0;
  let rejectedCount = 0;

  for (const file of files) {
    const validationError = validateFile(file);
    if (validationError) {
      rejectedCount += 1;
      toast("Skipped File", `${file.name}: ${validationError}`, "error");
      continue;
    }

    try {
      const queueItem = await createQueueItem(file);
      state.queue.push(queueItem);
      addedCount += 1;

      if (!state.selectedId) {
        state.selectedId = queueItem.id;
      }
    } catch (error) {
      rejectedCount += 1;
      console.error(error);
      toast("Load Failed", `Could not read ${file.name}.`, "error");
    }
  }

  if (addedCount > 0) {
    updateStatus(`${addedCount} image${addedCount > 1 ? "s" : ""} added to the queue.`, "success");
  } else if (rejectedCount > 0) {
    updateStatus("No valid images were added.", "error");
  }

  render();
}

async function createQueueItem(file) {
  const sourceUrl = URL.createObjectURL(file);
  const image = await loadImageFromUrl(sourceUrl);
  return {
    id: createId(),
    file,
    name: file.name,
    type: file.type,
    size: file.size,
    width: image.width,
    height: image.height,
    sourceUrl,
    resultBlob: null,
    resultUrl: "",
    resultWidth: 0,
    resultHeight: 0,
    status: "queued",
    progress: 0,
    error: "",
    processingTimeMs: 0,
  };
}

function validateFile(file) {
  if (!SUPPORTED_TYPES.includes(file.type)) {
    return "Unsupported file type.";
  }

  if (file.size > MAX_FILE_SIZE) {
    return "File exceeds 10MB.";
  }

  return "";
}

async function processBatch() {
  const processableItems = state.queue.filter((item) => item.status === "queued" || item.status === "failed");
  if (processableItems.length === 0 || state.isBatchProcessing) {
    return;
  }

  state.isBatchProcessing = true;
  state.cancelRequested = false;
  state.activeJobId += 1;
  updateStatus(`Starting batch for ${processableItems.length} image${processableItems.length > 1 ? "s" : ""}.`);
  render();

  const batchJobId = state.activeJobId;

  for (const item of processableItems) {
    if (state.cancelRequested || batchJobId !== state.activeJobId) {
      item.status = "cancelled";
      item.progress = 0;
      continue;
    }

    try {
      await processQueueItem(item, batchJobId);
    } catch (error) {
      if (error.message === "Cancelled") {
        item.status = "cancelled";
        item.progress = 0;
      } else {
        console.error(error);
        item.status = "failed";
        item.error = error.message || "Unexpected processing error.";
        item.progress = 0;
      }
    }

    render();
    await pauseForUi();
  }

  state.isBatchProcessing = false;
  state.cancelRequested = false;
  updateStatus(buildBatchSummaryMessage(), hasFailures() ? "error" : "success");
  render();
}

function cancelBatch() {
  if (!state.isBatchProcessing) {
    return;
  }

  state.cancelRequested = true;
  state.activeJobId += 1;
  updateStatus("Cancellation requested. The batch will stop as soon as the active item yields control.", "error");
  render();
}

async function processQueueItem(item, jobId) {
  const startedAt = performance.now();
  item.status = "processing";
  item.progress = 2;
  item.error = "";
  selectItem(item.id);
  updateStatus(`Processing ${item.name}...`);
  updatePreviewProgress(item.progress, "Preparing enhancement pipeline...");
  render();

  const sourceImage = await loadImageFromUrl(item.sourceUrl);
  ensureJobActive(jobId);

  const normalized = await normalizeLargeSource(sourceImage, item.type, jobId);
  const options = getProcessingOptions(normalized.width, normalized.height);
  let imageData;
  let resultWidth;
  let resultHeight;
  let appliedScale;

  const aiResult = await tryAiUpscale(normalized.image, options, item, jobId);

  if (aiResult) {
    imageData = aiResult.imageData;
    resultWidth = aiResult.width;
    resultHeight = aiResult.height;
    appliedScale = aiResult.appliedScale;
  } else {
    const scaleResult = await progressiveScaleImage(normalized.image, options, item, jobId);
    imageData = await enhanceImageData(scaleResult.imageData, scaleResult.width, scaleResult.height, options, item, jobId);
    resultWidth = scaleResult.width;
    resultHeight = scaleResult.height;
    appliedScale = scaleResult.appliedScale;
  }

  const resultBlob = await exportProcessedResult(imageData, resultWidth, resultHeight);
  ensureJobActive(jobId);

  cleanupItemResult(item);
  item.resultBlob = resultBlob;
  item.resultUrl = URL.createObjectURL(resultBlob);
  item.resultWidth = resultWidth;
  item.resultHeight = resultHeight;
  item.status = "complete";
  item.progress = 100;
  item.processingTimeMs = performance.now() - startedAt;
  item.appliedScale = appliedScale;

  updatePreviewProgress(100, "Enhancement complete.");
  render();
}

function getProcessingOptions(sourceWidth, sourceHeight) {
  const mode = elements.modeSelect.value;
  const requestedScale = Number(elements.scaleSelect.value);
  const sharpnessStrength = Number(elements.sharpnessSlider.value) / 100;
  const quality = Number(elements.qualitySlider.value) / 100;
  const modeConfig = MODE_CONFIG[mode];
  const aiModel = AI_MODEL_CONFIG[mode];
  const outputScale = getSafeOutputScale(sourceWidth, sourceHeight, requestedScale);

  return {
    mode,
    requestedScale,
    outputScale,
    quality,
    sharpnessStrength,
    aiModel,
    ...modeConfig,
  };
}

async function tryAiUpscale(image, options, item, jobId) {
  if (!shouldUseAiPath(image, options)) {
    elements.aiStatus.textContent = "Classic fallback";
    return null;
  }

  try {
    const sourceImageData = await getImageDataFromImage(image, jobId);
    const response = await callAiWorker(
      {
        type: "upscale",
        payload: {
          modelUrl: options.aiModel.url,
          runtimeUrl: ORT_WEBGPU_CDN,
          requestedScale: options.requestedScale,
          modelScale: AI_MODEL_SCALE,
          aiProfile: options.aiModel.profile,
          width: sourceImageData.width,
          height: sourceImageData.height,
          tileSize: getAiTileSize(image, options),
          tileOverlap: AI_TILE_OVERLAP,
          preferGpu: Boolean(navigator.gpu),
          outputScale: options.outputScale,
          options: {
            denoiseStrength: options.denoiseBias,
            sharpnessStrength: options.sharpnessStrength,
          },
          buffer: sourceImageData.data.buffer,
        },
      },
      [sourceImageData.data.buffer],
      jobId,
      item.id
    );

    ensureJobActive(jobId);
    elements.aiStatus.textContent = response.providerLabel || "AI ready";
    return {
      imageData: new ImageData(new Uint8ClampedArray(response.buffer), response.width, response.height),
      width: response.width,
      height: response.height,
      appliedScale: response.appliedScale,
    };
  } catch (error) {
    console.error(error);
    elements.aiStatus.textContent = "Classic fallback";
    return null;
  }
}

function shouldUseAiPath(image, options) {
  if (!state.aiWorker) {
    return false;
  }

  const lowMemory = isLowMemoryDevice();
  const sourcePixels = image.width * image.height;

  if (lowMemory && sourcePixels > 6_000_000) {
    return false;
  }

  return Boolean(options.aiModel?.url);
}

function getAiTileSize(image, options) {
  if (isLowMemoryDevice() || image.width * image.height > 8_000_000) {
    return 128;
  }

  return options.aiModel.tileSize;
}

async function progressiveScaleImage(image, options, item, jobId) {
  const appliedScale = options.outputScale;
  const targetWidth = Math.max(1, Math.round(image.width * appliedScale));
  const targetHeight = Math.max(1, Math.round(image.height * appliedScale));
  const steps = getProgressiveScaleSteps(appliedScale);

  let currentCanvas = document.createElement("canvas");
  let currentContext = currentCanvas.getContext("2d", { alpha: true });
  currentCanvas.width = image.width;
  currentCanvas.height = image.height;
  currentContext.drawImage(image, 0, 0);

  updateItemProgress(item, 8, "Analyzing source image...");

  for (let index = 0; index < steps.length; index += 1) {
    ensureJobActive(jobId);

    const stepScale = steps.slice(0, index + 1).reduce((total, value) => total * value, 1);
    const stepWidth = index === steps.length - 1 ? targetWidth : Math.max(1, Math.round(image.width * stepScale));
    const stepHeight = index === steps.length - 1 ? targetHeight : Math.max(1, Math.round(image.height * stepScale));

    const nextCanvas = document.createElement("canvas");
    const nextContext = nextCanvas.getContext("2d", { alpha: true });
    nextCanvas.width = stepWidth;
    nextCanvas.height = stepHeight;
    nextContext.imageSmoothingEnabled = true;
    nextContext.imageSmoothingQuality = "high";
    nextContext.drawImage(currentCanvas, 0, 0, stepWidth, stepHeight);

    currentCanvas.width = 1;
    currentCanvas.height = 1;
    currentCanvas = nextCanvas;
    currentContext = nextContext;

    updateItemProgress(item, 16 + Math.round(((index + 1) / steps.length) * 26), `Scaling pass ${index + 1} of ${steps.length}...`);
    await yieldToBrowser(jobId);
  }

  const imageData = currentContext.getImageData(0, 0, currentCanvas.width, currentCanvas.height);
  currentCanvas.width = 1;
  currentCanvas.height = 1;

  return {
    imageData,
    width: targetWidth,
    height: targetHeight,
    appliedScale,
  };
}

async function enhanceImageData(imageData, width, height, options, item, jobId) {
  ensureJobActive(jobId);
  updateItemProgress(item, 56, "Applying clarity enhancement...");

  const response = await callPixelWorker(
    {
      type: "enhance",
      payload: {
        width,
        height,
        options,
        buffer: imageData.data.buffer,
      },
    },
    [imageData.data.buffer],
    jobId
  );

  ensureJobActive(jobId);
  updateItemProgress(item, 88, "Rendering export output...");

  return new ImageData(new Uint8ClampedArray(response.buffer), width, height);
}

async function exportProcessedResult(imageData, width, height) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { alpha: true });
  canvas.width = width;
  canvas.height = height;
  context.putImageData(imageData, 0, 0);

  const blob = await canvasToBlob(
    canvas,
    elements.formatSelect.value,
    Number(elements.qualitySlider.value) / 100
  );

  canvas.width = 1;
  canvas.height = 1;
  return blob;
}

function updateItemProgress(item, progress, stage) {
  item.progress = progress;
  if (state.selectedId === item.id) {
    updatePreviewProgress(progress, stage);
  }
  render();
}

function updatePreviewProgress(progress, stage) {
  const clamped = clamp(Math.round(progress), 0, 100);
  elements.progressValue.textContent = `${clamped}%`;
  elements.progressStage.textContent = stage;
}

async function normalizeLargeSource(image, mimeType, jobId) {
  const totalPixels = image.width * image.height;
  if (totalPixels <= MAX_SOURCE_PIXELS) {
    return {
      image,
      width: image.width,
      height: image.height,
    };
  }

  const scale = Math.sqrt(MAX_SOURCE_PIXELS / totalPixels);
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { alpha: true });
  canvas.width = width;
  canvas.height = height;
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, 0, 0, width, height);

  await yieldToBrowser(jobId);

  const blob = await canvasToBlob(canvas, mimeType, 0.96);
  const normalizedUrl = URL.createObjectURL(blob);
  const normalizedImage = await loadImageFromUrl(normalizedUrl);

  canvas.width = 1;
  canvas.height = 1;
  URL.revokeObjectURL(normalizedUrl);

  return {
    image: normalizedImage,
    width,
    height,
  };
}

function render() {
  renderQueue();
  renderCounters();
  renderSelectedPreview();
  renderActionState();
}

function renderQueue() {
  const items = state.queue;
  elements.queueGrid.innerHTML = "";
  elements.emptyQueue.hidden = items.length > 0;

  for (const item of items) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `queue-card${item.id === state.selectedId ? " is-selected" : ""}`;
    card.addEventListener("click", () => selectItem(item.id));

    const statusLabel = getStatusLabel(item.status);
    const resultMeta =
      item.status === "complete"
        ? `${item.resultWidth} x ${item.resultHeight}`
        : `${item.width} x ${item.height}`;

    card.innerHTML = `
      <div class="queue-thumb">
        <img src="${escapeHtml(item.sourceUrl)}" alt="${escapeHtml(item.name)} thumbnail" />
      </div>
      <div class="queue-card-head">
        <div>
          <h3 class="queue-card-title">${escapeHtml(item.name)}</h3>
          <span class="queue-card-meta">${resultMeta}</span>
          <span class="queue-card-sub">${formatFileSize(item.size)} • ${item.type.replace("image/", "").toUpperCase()}</span>
        </div>
        <span class="queue-status status-${item.status}">${statusLabel}</span>
      </div>
      <div class="queue-progress">
        <div class="progress-bar">
          <div class="progress-fill" style="width:${clamp(item.progress, 0, 100)}%"></div>
        </div>
        <span class="queue-card-sub">${item.error ? escapeHtml(item.error) : `${Math.round(item.progress)}%`}</span>
      </div>
    `;

    elements.queueGrid.appendChild(card);
  }
}

function renderCounters() {
  const counts = getCounts();
  const total = state.queue.length;
  const finished = counts.complete + counts.failed;
  const progress = total > 0 ? Math.round((finished / total) * 100) : 0;

  elements.queueCount.textContent = String(total);
  elements.completedCount.textContent = String(counts.complete);
  elements.failedCount.textContent = String(counts.failed);
  elements.pendingCount.textContent = String(counts.queued + counts.cancelled);
  elements.processingCount.textContent = String(counts.processing);
  elements.doneCount.textContent = String(counts.complete);
  elements.batchFailedCount.textContent = String(counts.failed);
  elements.batchProgressText.textContent = `${progress}%`;
  elements.batchProgressFill.style.width = `${progress}%`;
  elements.batchProgressLabel.textContent = state.isBatchProcessing
    ? "Batch in progress"
    : total > 0
      ? "Batch ready"
      : "Batch progress";
}

function renderSelectedPreview() {
  const item = getSelectedItem();

  if (!item) {
    elements.selectedTitle.textContent = "No image selected";
    elements.selectedMeta.textContent = "Choose a queue item to inspect.";
    hideImage(elements.originalPreview, elements.originalEmpty);
    hideImage(elements.resultPreview, elements.resultEmpty);
    hideCompare();
    elements.originalMeta.textContent = "Waiting for upload";
    elements.resultMeta.textContent = "No result yet";
    return;
  }

  elements.selectedTitle.textContent = item.name;
  elements.selectedMeta.textContent = `${item.width} x ${item.height} • ${formatFileSize(item.size)} • ${getStatusLabel(item.status)}`;

  showImage(elements.originalPreview, elements.originalEmpty, item.sourceUrl);
  elements.originalMeta.textContent = `${item.width} x ${item.height} • ${item.type.replace("image/", "").toUpperCase()}`;

  if (item.resultUrl) {
    showImage(elements.resultPreview, elements.resultEmpty, item.resultUrl);
    elements.resultMeta.textContent = `${item.resultWidth} x ${item.resultHeight} • ${getFormatLabel(elements.formatSelect.value)}`;
    showCompare(item);
  } else {
    hideImage(elements.resultPreview, elements.resultEmpty);
    elements.resultMeta.textContent = item.status === "processing" ? "Processing..." : "No result yet";
    hideCompare();
  }

  elements.loader.hidden = !(state.isBatchProcessing && item.status === "processing");
}

function renderActionState() {
  const hasQueue = state.queue.length > 0;
  const hasCompleted = state.queue.some((item) => item.status === "complete");
  const hasProcessable = state.queue.some((item) => item.status === "queued" || item.status === "failed");
  const selected = getSelectedItem();
  const canDownloadSelected = Boolean(selected?.resultUrl);

  elements.upscaleAllBtn.disabled = state.isBatchProcessing || !hasProcessable;
  elements.cancelBtn.disabled = !state.isBatchProcessing;
  elements.zipBtn.disabled = state.isBatchProcessing || !hasCompleted;
  elements.clearCompletedBtn.disabled = state.isBatchProcessing || !hasCompleted;
  elements.resetAllBtn.disabled = state.isBatchProcessing || !hasQueue;
  elements.downloadSelectedBtn.disabled = !canDownloadSelected || state.isBatchProcessing;
}

function selectItem(id) {
  state.selectedId = id;
  render();
}

function getSelectedItem() {
  return state.queue.find((item) => item.id === state.selectedId) || null;
}

function clearCompletedItems() {
  if (state.isBatchProcessing) {
    return;
  }

  const retained = [];
  for (const item of state.queue) {
    if (item.status === "complete") {
      cleanupItem(item);
      continue;
    }
    retained.push(item);
  }

  state.queue = retained;
  ensureSelectedItem();
  updateStatus("Completed items removed from the queue.", "success");
  render();
}

function resetQueue() {
  if (state.isBatchProcessing) {
    return;
  }

  for (const item of state.queue) {
    cleanupItem(item);
  }

  state.queue = [];
  state.selectedId = "";
  updateStatus("Queue reset. Add images to start another batch.");
  updatePreviewProgress(0, "Preparing enhancement pipeline...");
  render();
}

function ensureSelectedItem() {
  if (state.queue.some((item) => item.id === state.selectedId)) {
    return;
  }

  state.selectedId = state.queue[0]?.id || "";
}

function downloadSelectedResult() {
  const item = getSelectedItem();
  if (!item?.resultBlob || !item.resultUrl) {
    return;
  }

  downloadBlob(item.resultUrl, buildResultFilename(item));
  toast("Download Started", `Exporting ${buildResultFilename(item)}`, "success");
}

async function downloadAllAsZip() {
  const completedItems = state.queue.filter((item) => item.status === "complete" && item.resultBlob);
  if (completedItems.length === 0) {
    toast("No Results Yet", "Complete at least one image before exporting a ZIP.", "error");
    return;
  }

  updateStatus(`Packaging ${completedItems.length} result${completedItems.length > 1 ? "s" : ""} into ZIP...`);

  const files = await Promise.all(
    completedItems.map(async (item) => ({
      name: buildResultFilename(item),
      bytes: new Uint8Array(await item.resultBlob.arrayBuffer()),
    }))
  );

  const zipBlob = createZipArchive(files);
  const zipUrl = URL.createObjectURL(zipBlob);
  downloadBlob(zipUrl, "sat-upscaled-batch.zip");
  URL.revokeObjectURL(zipUrl);
  toast("ZIP Ready", "Batch archive download started.", "success");
}

function buildResultFilename(item) {
  const extension = getExtensionFromType(elements.formatSelect.value);
  const baseName = stripExtension(item.name);
  return `${sanitizeFilename(baseName)}_upscaled.${extension}`;
}

function showImage(image, emptyState, src) {
  image.src = src;
  image.classList.add("visible");
  emptyState.hidden = true;
}

function hideImage(image, emptyState) {
  image.removeAttribute("src");
  image.classList.remove("visible");
  emptyState.hidden = false;
}

function showCompare(item) {
  elements.compareBase.src = item.sourceUrl;
  elements.compareEnhanced.src = item.resultUrl;
  elements.compareBase.classList.add("visible");
  elements.compareEnhanced.classList.add("visible");
  elements.compareOverlay.classList.add("visible");
  elements.compareEmpty.hidden = true;
  elements.compareSlider.disabled = false;
  updateComparison(state.compareValue);
}

function hideCompare() {
  elements.compareBase.removeAttribute("src");
  elements.compareEnhanced.removeAttribute("src");
  elements.compareBase.classList.remove("visible");
  elements.compareEnhanced.classList.remove("visible");
  elements.compareOverlay.classList.remove("visible");
  elements.compareEmpty.hidden = false;
  elements.compareSlider.disabled = true;
}

function updateComparison(value) {
  state.compareValue = clamp(value, 0, 100);
  elements.compareSlider.value = String(state.compareValue);
  elements.compareOverlay.style.width = `${state.compareValue}%`;
  elements.compareLine.style.left = `${state.compareValue}%`;
}

function syncControlLabels() {
  elements.qualityValue.textContent = `${elements.qualitySlider.value}%`;
  elements.sharpnessValue.textContent = `${elements.sharpnessSlider.value}%`;
}

function updateSafeModeBadge() {
  elements.safeModeBadge.textContent = isLowMemoryDevice() ? "Safe mode: on" : "Safe mode: auto";
}

function updateStatus(message, type = "") {
  elements.statusMessage.textContent = message;
  elements.statusMessage.className = "status-message";
  if (type) {
    elements.statusMessage.classList.add(type);
  }
}

function getCounts() {
  return state.queue.reduce(
    (counts, item) => {
      counts[item.status] = (counts[item.status] || 0) + 1;
      return counts;
    },
    { queued: 0, processing: 0, complete: 0, failed: 0, cancelled: 0 }
  );
}

function buildBatchSummaryMessage() {
  const counts = getCounts();
  return `${counts.complete} completed, ${counts.failed} failed, ${counts.cancelled} cancelled.`;
}

function hasFailures() {
  return state.queue.some((item) => item.status === "failed");
}

function getStatusLabel(status) {
  switch (status) {
    case "processing":
      return "Processing";
    case "complete":
      return "Done";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    default:
      return "Queued";
  }
}

function getFormatLabel(type) {
  return type.replace("image/", "").toUpperCase();
}

function ensureJobActive(jobId) {
  if (jobId !== state.activeJobId || state.cancelRequested) {
    throw new Error("Cancelled");
  }
}

async function yieldToBrowser(jobId) {
  ensureJobActive(jobId);
  await pauseForUi();
  ensureJobActive(jobId);
}

async function pauseForUi() {
  await new Promise((resolve) => {
    requestAnimationFrame(() => setTimeout(resolve, 0));
  });
}

function isLowMemoryDevice() {
  return Boolean(navigator.deviceMemory && navigator.deviceMemory <= 4);
}

async function getImageDataFromImage(image, jobId) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { alpha: true, willReadFrequently: true });
  canvas.width = image.width;
  canvas.height = image.height;
  context.drawImage(image, 0, 0);

  await yieldToBrowser(jobId);

  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  canvas.width = 1;
  canvas.height = 1;
  return imageData;
}

function getSafeOutputScale(width, height, requestedScale) {
  const requestedPixels = width * requestedScale * height * requestedScale;
  if (requestedPixels <= MAX_OUTPUT_PIXELS) {
    return requestedScale;
  }
  return Math.sqrt(MAX_OUTPUT_PIXELS / (width * height));
}

function getProgressiveScaleSteps(scaleFactor) {
  if (scaleFactor >= 3.2) {
    return [1.42, 1.42, scaleFactor / (1.42 * 1.42)];
  }
  return [1.35, scaleFactor / 1.35];
}

function ensureAiWorker() {
  if (!("Worker" in window)) {
    elements.aiStatus.textContent = "Worker unavailable";
    return;
  }

  try {
    state.aiWorker = new Worker("./ai-upscale-worker.js");
    state.aiWorker.addEventListener("message", handleAiWorkerMessage);
    state.aiWorker.addEventListener("error", () => {
      state.aiWorker?.terminate();
      state.aiWorker = null;
      elements.aiStatus.textContent = "AI worker failed";
    });
  } catch (error) {
    console.error(error);
    state.aiWorker = null;
    elements.aiStatus.textContent = "AI worker failed";
  }
}

function handleAiWorkerMessage(event) {
  const { id, type, itemId } = event.data;
  if (type === "progress") {
    handleAiProgress(event.data);
    return;
  }

  const resolver = state.pendingAiResolvers.get(id);
  if (!resolver) {
    return;
  }

  state.pendingAiResolvers.delete(id);

  if (type === "error") {
    resolver.reject(new Error(event.data.message || "AI worker error"));
    return;
  }

  resolver.resolve({ ...event.data, itemId });
}

function handleAiProgress(message) {
  const item = state.queue.find((entry) => entry.id === message.itemId);
  if (!item || item.status !== "processing") {
    return;
  }

  switch (message.phase) {
    case "runtime-loading":
      elements.aiStatus.textContent = "Loading runtime";
      updateItemProgress(item, 10, "Loading ONNX runtime...");
      break;
    case "model-download":
      elements.aiStatus.textContent = "Downloading model";
      updateItemProgress(item, message.total > 0 ? 12 + Math.round((message.loaded / message.total) * 18) : 20, "Downloading AI model...");
      break;
    case "session-creating":
      elements.aiStatus.textContent = "Creating session";
      updateItemProgress(item, 34, "Creating AI inference session...");
      break;
    case "session-ready":
      elements.aiStatus.textContent = message.providerLabel || "AI ready";
      updateItemProgress(item, 42, `AI session ready (${message.providerLabel || "wasm"})`);
      break;
    case "tiling":
      elements.aiStatus.textContent = "Preparing tiles";
      updateItemProgress(item, 48, "Preparing tiled inference...");
      break;
    case "tile":
      elements.aiStatus.textContent = "Inferencing";
      updateItemProgress(item, 48 + Math.round((message.completed / message.total) * 40), `Running tile ${message.completed} of ${message.total}...`);
      break;
    case "downscale":
      elements.aiStatus.textContent = "Finishing";
      updateItemProgress(item, 92, "Finalizing scaled output...");
      break;
    default:
      break;
  }
}

function callAiWorker(message, transfer, jobId, itemId) {
  return new Promise((resolve, reject) => {
    if (!state.aiWorker) {
      reject(new Error("AI worker unavailable"));
      return;
    }

    const id = `ai-${jobId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    state.pendingAiResolvers.set(id, { resolve, reject });
    state.aiWorker.postMessage({ ...message, id, itemId }, transfer);
  });
}

function ensurePixelWorker() {
  if (!("Worker" in window)) {
    return;
  }

  const workerSource = `
    self.onmessage = (event) => {
      const { id, type, payload } = event.data;
      if (type !== "enhance") {
        return;
      }

      try {
        const source = new Uint8ClampedArray(payload.buffer);
        const result = enhance(source, payload.width, payload.height, payload.options);
        self.postMessage({ id, buffer: result.buffer }, [result.buffer]);
      } catch (error) {
        self.postMessage({ id, type: "error", message: error.message || "Enhancement worker failed" });
      }
    };

    function enhance(source, width, height, options) {
      const denoised = denoise(source, width, height, options);
      const blurred = blur(denoised, width, height);
      const edges = edgeMap(denoised, width, height);
      const output = new Uint8ClampedArray(source.length);

      for (let index = 0; index < source.length; index += 4) {
        const edgeStrength = edges[index] / 255;
        const textMask = edgeStrength > 0.24 ? 1 : 0;

        for (let channel = 0; channel < 3; channel += 1) {
          const original = denoised[index + channel];
          const blurValue = blurred[index + channel];
          const detail = original - blurValue;
          const edgeBoost = detail * options.edgeBoost * edgeStrength * (1 + options.sharpnessStrength);
          const detailBoost = detail * options.detailBoost;
          const textBoost = detail * options.textBoost * textMask;
          const saturationOffset = (original - 128) * options.saturationBoost * 0.08;
          output[index + channel] = clampColor(original + edgeBoost + detailBoost + textBoost + saturationOffset);
        }

        output[index + 3] = denoised[index + 3];
      }

      return output;
    }

    function denoise(source, width, height, options) {
      const blurred = blur(source, width, height);
      const output = new Uint8ClampedArray(source.length);
      const mix = Math.min(0.42, options.denoiseBias + 0.12);

      for (let index = 0; index < source.length; index += 4) {
        for (let channel = 0; channel < 3; channel += 1) {
          output[index + channel] = clampColor(source[index + channel] * (1 - mix) + blurred[index + channel] * mix);
        }
        output[index + 3] = source[index + 3];
      }

      return output;
    }

    function blur(source, width, height) {
      const output = new Uint8ClampedArray(source.length);

      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          let totalRed = 0;
          let totalGreen = 0;
          let totalBlue = 0;
          let totalAlpha = 0;
          let totalWeight = 0;

          for (let oy = -1; oy <= 1; oy += 1) {
            for (let ox = -1; ox <= 1; ox += 1) {
              const px = clampIndex(x + ox, 0, width - 1);
              const py = clampIndex(y + oy, 0, height - 1);
              const index = (py * width + px) * 4;
              const weight = ox === 0 && oy === 0 ? 2 : 1;
              totalRed += source[index] * weight;
              totalGreen += source[index + 1] * weight;
              totalBlue += source[index + 2] * weight;
              totalAlpha += source[index + 3] * weight;
              totalWeight += weight;
            }
          }

          const target = (y * width + x) * 4;
          output[target] = Math.round(totalRed / totalWeight);
          output[target + 1] = Math.round(totalGreen / totalWeight);
          output[target + 2] = Math.round(totalBlue / totalWeight);
          output[target + 3] = Math.round(totalAlpha / totalWeight);
        }
      }

      return output;
    }

    function edgeMap(source, width, height) {
      const output = new Uint8ClampedArray(source.length);

      for (let y = 1; y < height - 1; y += 1) {
        for (let x = 1; x < width - 1; x += 1) {
          const index = (y * width + x) * 4;
          const left = intensity(source, ((y * width) + (x - 1)) * 4);
          const right = intensity(source, ((y * width) + (x + 1)) * 4);
          const top = intensity(source, (((y - 1) * width) + x) * 4);
          const bottom = intensity(source, (((y + 1) * width) + x) * 4);
          const value = Math.min(255, Math.abs(right - left) + Math.abs(bottom - top));
          output[index] = value;
          output[index + 1] = value;
          output[index + 2] = value;
          output[index + 3] = source[index + 3];
        }
      }

      return output;
    }

    function intensity(source, index) {
      return source[index] * 0.299 + source[index + 1] * 0.587 + source[index + 2] * 0.114;
    }

    function clampColor(value) {
      return Math.max(0, Math.min(255, Math.round(value)));
    }

    function clampIndex(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }
  `;

  const blob = new Blob([workerSource], { type: "application/javascript" });
  const workerUrl = URL.createObjectURL(blob);
  state.pixelWorker = new Worker(workerUrl);
  URL.revokeObjectURL(workerUrl);

  state.pixelWorker.addEventListener("message", (event) => {
    const { id, type } = event.data;
    const resolver = state.pendingPixelResolvers.get(id);
    if (!resolver) {
      return;
    }

    state.pendingPixelResolvers.delete(id);

    if (type === "error") {
      resolver.reject(new Error(event.data.message || "Enhancement worker error"));
      return;
    }

    resolver.resolve(event.data);
  });

  state.pixelWorker.addEventListener("error", () => {
    state.pixelWorker?.terminate();
    state.pixelWorker = null;
  });
}

function callPixelWorker(message, transfer, jobId) {
  return new Promise((resolve, reject) => {
    if (!state.pixelWorker) {
      reject(new Error("Enhancement worker unavailable"));
      return;
    }

    const id = `px-${jobId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    state.pendingPixelResolvers.set(id, { resolve, reject });
    state.pixelWorker.postMessage({ ...message, id }, transfer);
  });
}

function getFilesFromClipboard(clipboardData) {
  if (!clipboardData?.items) {
    return [];
  }

  const files = [];
  for (const item of clipboardData.items) {
    if (item.kind === "file" && SUPPORTED_TYPES.includes(item.type)) {
      const file = item.getAsFile();
      if (file) {
        files.push(file);
      }
    }
  }
  return files;
}

function canvasToBlob(canvas, mimeType, quality) {
  const safeMimeType = SUPPORTED_TYPES.includes(mimeType) ? mimeType : "image/jpeg";

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Canvas export failed."));
          return;
        }
        resolve(blob);
      },
      safeMimeType,
      quality
    );
  });
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

function cleanupItemResult(item) {
  if (item.resultUrl) {
    URL.revokeObjectURL(item.resultUrl);
    item.resultUrl = "";
  }
  item.resultBlob = null;
}

function cleanupItem(item) {
  if (item.sourceUrl) {
    URL.revokeObjectURL(item.sourceUrl);
  }
  cleanupItemResult(item);
}

function downloadBlob(url, filename) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
}

function toast(title, message, type = "success") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.innerHTML = `<strong>${escapeHtml(title)}</strong><p>${escapeHtml(message)}</p>`;
  elements.toastStack.appendChild(item);

  setTimeout(() => {
    item.remove();
  }, TOAST_DURATION);
}

function createZipArchive(files) {
  const encoder = new TextEncoder();
  const localParts = [];
  const centralParts = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const crc = crc32(file.bytes);
    const localHeader = new Uint8Array(30 + nameBytes.length);
    const localView = new DataView(localHeader.buffer);

    localView.setUint32(0, 0x04034b50, true);
    localView.setUint16(4, 20, true);
    localView.setUint16(6, 0, true);
    localView.setUint16(8, 0, true);
    localView.setUint16(10, 0, true);
    localView.setUint16(12, 0, true);
    localView.setUint32(14, crc, true);
    localView.setUint32(18, file.bytes.length, true);
    localView.setUint32(22, file.bytes.length, true);
    localView.setUint16(26, nameBytes.length, true);
    localView.setUint16(28, 0, true);
    localHeader.set(nameBytes, 30);

    localParts.push(localHeader, file.bytes);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const centralView = new DataView(centralHeader.buffer);
    centralView.setUint32(0, 0x02014b50, true);
    centralView.setUint16(4, 20, true);
    centralView.setUint16(6, 20, true);
    centralView.setUint16(8, 0, true);
    centralView.setUint16(10, 0, true);
    centralView.setUint16(12, 0, true);
    centralView.setUint16(14, 0, true);
    centralView.setUint32(16, crc, true);
    centralView.setUint32(20, file.bytes.length, true);
    centralView.setUint32(24, file.bytes.length, true);
    centralView.setUint16(28, nameBytes.length, true);
    centralView.setUint16(30, 0, true);
    centralView.setUint16(32, 0, true);
    centralView.setUint16(34, 0, true);
    centralView.setUint16(36, 0, true);
    centralView.setUint32(38, 0, true);
    centralView.setUint32(42, offset, true);
    centralHeader.set(nameBytes, 46);
    centralParts.push(centralHeader);

    offset += localHeader.length + file.bytes.length;
  }

  const centralSize = centralParts.reduce((total, part) => total + part.length, 0);
  const endRecord = new Uint8Array(22);
  const endView = new DataView(endRecord.buffer);
  endView.setUint32(0, 0x06054b50, true);
  endView.setUint16(4, 0, true);
  endView.setUint16(6, 0, true);
  endView.setUint16(8, files.length, true);
  endView.setUint16(10, files.length, true);
  endView.setUint32(12, centralSize, true);
  endView.setUint32(16, offset, true);
  endView.setUint16(20, 0, true);

  return new Blob([...localParts, ...centralParts, endRecord], { type: "application/zip" });
}

function crc32(bytes) {
  let crc = -1;
  for (let index = 0; index < bytes.length; index += 1) {
    crc = (crc >>> 8) ^ CRC32_TABLE[(crc ^ bytes[index]) & 0xff];
  }
  return (crc ^ -1) >>> 0;
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) ? (0xedb88320 ^ (value >>> 1)) : (value >>> 1);
    }
    table[index] = value >>> 0;
  }
  return table;
})();

function getExtensionFromType(type) {
  switch (type) {
    case "image/png":
      return "png";
    case "image/webp":
      return "webp";
    default:
      return "jpg";
  }
}

function stripExtension(name) {
  return name.replace(/\.[^.]+$/, "");
}

function sanitizeFilename(name) {
  return name.replace(/[<>:"/\\|?*\x00-\x1F]/g, "-").replace(/\s+/g, "-");
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function createId() {
  return `item-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}
