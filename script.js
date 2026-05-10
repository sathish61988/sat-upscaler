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
  general: {
    label: "General",
    url: "./models/realesrgan-general-x4.onnx",
    tileSize: 192,
  },
  portrait: {
    label: "Portrait",
    url: "./models/realesrgan-portrait-x4.onnx",
    tileSize: 192,
  },
  anime: {
    label: "Anime",
    url: "./models/realesrgan-anime-x4.onnx",
    tileSize: 224,
  },
};

const MODE_CONFIG = {
  balanced: {
    denoiseBias: 0.18,
    edgeBoost: 0.46,
    detailBoost: 0.18,
    textBoost: 0.18,
    saturationBoost: 0.06,
  },
  sharp: {
    denoiseBias: 0.1,
    edgeBoost: 0.74,
    detailBoost: 0.28,
    textBoost: 0.34,
    saturationBoost: 0.08,
  },
  photo: {
    denoiseBias: 0.24,
    edgeBoost: 0.38,
    detailBoost: 0.16,
    textBoost: 0.12,
    saturationBoost: 0.1,
  },
  anime: {
    denoiseBias: 0.12,
    edgeBoost: 0.7,
    detailBoost: 0.3,
    textBoost: 0.2,
    saturationBoost: 0.12,
  },
};

const elements = {
  fileInput: document.getElementById("fileInput"),
  heroUploadBtn: document.getElementById("heroUploadBtn"),
  ctaUploadBtn: document.getElementById("ctaUploadBtn"),
  dropzone: document.getElementById("dropzone"),
  statusMessage: document.getElementById("statusMessage"),
  timeEstimate: document.getElementById("timeEstimate"),
  outputPreview: document.getElementById("outputPreview"),
  modeSelect: document.getElementById("modeSelect"),
  aiStatus: document.getElementById("aiStatus"),
  aiProfileSelect: document.getElementById("aiProfileSelect"),
  aiHint: document.getElementById("aiHint"),
  denoiseSlider: document.getElementById("denoiseSlider"),
  denoiseValue: document.getElementById("denoiseValue"),
  sharpnessSlider: document.getElementById("sharpnessSlider"),
  sharpnessValue: document.getElementById("sharpnessValue"),
  formatSelect: document.getElementById("formatSelect"),
  qualitySlider: document.getElementById("qualitySlider"),
  qualityValue: document.getElementById("qualityValue"),
  filenameInput: document.getElementById("filenameInput"),
  upscale2xBtn: document.getElementById("upscale2xBtn"),
  upscale4xBtn: document.getElementById("upscale4xBtn"),
  downloadBtn: document.getElementById("downloadBtn"),
  resetBtn: document.getElementById("resetBtn"),
  fullscreenBtn: document.getElementById("fullscreenBtn"),
  originalPreview: document.getElementById("originalPreview"),
  upscaledPreview: document.getElementById("upscaledPreview"),
  originalCanvasWrap: document.getElementById("originalCanvasWrap"),
  resultCanvasWrap: document.getElementById("resultCanvasWrap"),
  originalZoom: document.getElementById("originalZoom"),
  resultZoom: document.getElementById("resultZoom"),
  originalEmpty: document.getElementById("originalEmpty"),
  resultEmpty: document.getElementById("resultEmpty"),
  originalBadge: document.getElementById("originalBadge"),
  resultBadge: document.getElementById("resultBadge"),
  originalMeta: document.getElementById("originalMeta"),
  resultMeta: document.getElementById("resultMeta"),
  compareBase: document.getElementById("compareBase"),
  compareEnhanced: document.getElementById("compareEnhanced"),
  compareOverlay: document.getElementById("compareOverlay"),
  compareLine: document.getElementById("compareLine"),
  compareSlider: document.getElementById("compareSlider"),
  compareEmpty: document.getElementById("compareEmpty"),
  loader: document.getElementById("loader"),
  progressValue: document.getElementById("progressValue"),
  progressStage: document.getElementById("progressStage"),
  progressText: document.getElementById("progressText"),
  progressBarFill: document.getElementById("progressBarFill"),
  uploadBurst: document.getElementById("uploadBurst"),
  toastStack: document.getElementById("toastStack"),
  fullscreenModal: document.getElementById("fullscreenModal"),
  closeModalBtn: document.getElementById("closeModalBtn"),
  modalCompareBase: document.getElementById("modalCompareBase"),
  modalCompareEnhanced: document.getElementById("modalCompareEnhanced"),
  modalCompareOverlay: document.getElementById("modalCompareOverlay"),
  modalCompareLine: document.getElementById("modalCompareLine"),
  modalCompareSlider: document.getElementById("modalCompareSlider"),
};

const state = {
  sourceFile: null,
  sourceImage: null,
  sourceObjectUrl: "",
  resultObjectUrl: "",
  resultBlob: null,
  resultMeta: null,
  compareValue: DEFAULT_COMPARE_VALUE,
  isProcessing: false,
  worker: null,
  aiWorker: null,
  activeJobId: 0,
  pendingAiResolvers: new Map(),
  pendingWorkerResolvers: new Map(),
};

initialize();

function initialize() {
  bindEvents();
  ensureWorker();
  ensureAiWorker();
  syncControlLabels();
  updateComparison(DEFAULT_COMPARE_VALUE);
  updateModalComparison(DEFAULT_COMPARE_VALUE);
  updateStatus("Ready for your first image.");
  updateAiHint();
  updateEstimate();
  updateOutputPreview();
}

function bindEvents() {
  elements.fileInput.addEventListener("change", handleFileInputChange);
  elements.heroUploadBtn?.addEventListener("click", () => {
    elements.fileInput.click();
  });
  elements.ctaUploadBtn?.addEventListener("click", () => {
    elements.fileInput.click();
  });
  elements.upscale2xBtn.addEventListener("click", () => processUpscale(2));
  elements.upscale4xBtn.addEventListener("click", () => processUpscale(4));
  elements.downloadBtn.addEventListener("click", downloadResult);
  elements.resetBtn.addEventListener("click", resetWorkspace);
  elements.fullscreenBtn.addEventListener("click", openFullscreenModal);
  elements.closeModalBtn.addEventListener("click", closeFullscreenModal);
  elements.fullscreenModal.addEventListener("click", (event) => {
    if (event.target instanceof HTMLElement && event.target.dataset.closeModal === "true") {
      closeFullscreenModal();
    }
  });

  elements.compareSlider.addEventListener("input", (event) => {
    const value = Number(event.target.value);
    updateComparison(value);
    updateModalComparison(value);
  });

  elements.modalCompareSlider.addEventListener("input", (event) => {
    const value = Number(event.target.value);
    updateComparison(value);
    updateModalComparison(value);
  });

  elements.modeSelect.addEventListener("change", () => {
    updateEstimate();
    updateOutputPreview();
    toast("Mode Updated", `${capitalize(elements.modeSelect.value)} mode is active.`, "success");
  });

  elements.aiProfileSelect.addEventListener("change", () => {
    updateAiHint();
    updateEstimate();
  });

  elements.denoiseSlider.addEventListener("input", () => {
    syncControlLabels();
    updateEstimate();
  });

  elements.sharpnessSlider.addEventListener("input", () => {
    syncControlLabels();
    updateEstimate();
  });

  elements.formatSelect.addEventListener("change", updateOutputPreview);
  elements.qualitySlider.addEventListener("input", () => {
    syncControlLabels();
    updateOutputPreview();
  });

  elements.filenameInput.addEventListener("input", () => {
    if (!elements.filenameInput.value.trim()) {
      elements.filenameInput.value = "sat-upscaled";
    }
  });

  elements.originalZoom.addEventListener("input", () => {
    updateZoom(elements.originalPreview, Number(elements.originalZoom.value));
  });

  elements.resultZoom.addEventListener("input", () => {
    updateZoom(elements.upscaledPreview, Number(elements.resultZoom.value));
  });

  bindDropzone();
  bindClipboardPaste();
  bindKeyboardShortcuts();
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
    const [file] = event.dataTransfer?.files || [];
    if (file) {
      handleFile(file);
    }
  });
}

function bindClipboardPaste() {
  window.addEventListener("paste", (event) => {
    const file = getFileFromClipboard(event.clipboardData);
    if (!file) {
      return;
    }

    event.preventDefault();
    handleFile(file);
    toast("Image Pasted", "Clipboard image added to the workspace.", "success");
  });
}

function bindKeyboardShortcuts() {
  window.addEventListener("keydown", (event) => {
    if (isTypingTarget(event.target)) {
      return;
    }

    const key = event.key.toLowerCase();

    if (key === "2") {
      event.preventDefault();
      processUpscale(2);
      return;
    }

    if (key === "4") {
      event.preventDefault();
      processUpscale(4);
      return;
    }

    if (key === "d") {
      event.preventDefault();
      downloadResult();
      return;
    }

    if (key === "r") {
      event.preventDefault();
      resetWorkspace();
      return;
    }

    if (key === "f") {
      event.preventDefault();
      openFullscreenModal();
      return;
    }

    if (key === "escape") {
      closeFullscreenModal();
    }
  });
}

function handleFileInputChange(event) {
  const [file] = event.target.files || [];
  if (file) {
    handleFile(file);
  }
}

async function handleFile(file) {
  if (state.isProcessing) {
    toast("Processing Active", "Please wait for the current enhancement to finish.", "error");
    return;
  }

  clearResultState();

  const validationError = validateFile(file);
  if (validationError) {
    updateStatus(validationError, "error");
    toast("Upload Error", validationError, "error");
    elements.fileInput.value = "";
    return;
  }

  try {
    const loadedImage = await loadImageFromFile(file);
    const normalized = await normalizeLargeSource(loadedImage, file.type);

    state.sourceFile = file;
    state.sourceImage = normalized.image;
    updateSourcePreview(normalized);
    updateZoom(elements.originalPreview, Number(elements.originalZoom.value));
    updateSourceMeta(file, normalized);
    updateEstimate();
    updateOutputPreview();
    updateActionAvailability();
    pulseUploadAnimation();
    updateStatus("Image loaded. Choose 2x or 4x to start enhancing.", "success");
    setBadge(elements.originalBadge, "Loaded");
    toast("Upload Complete", "Image is ready for enhancement.", "success");
  } catch (error) {
    console.error(error);
    updateStatus("Could not open this image. Please try another file.", "error");
    toast("Load Failed", "The image could not be opened.", "error");
  }
}

function validateFile(file) {
  if (!SUPPORTED_TYPES.includes(file.type)) {
    return "Unsupported file type. Please use JPG, PNG, or WEBP.";
  }

  if (file.size > MAX_FILE_SIZE) {
    return "File is too large. Please upload an image under 10MB.";
  }

  return "";
}

async function loadImageFromFile(file) {
  cleanupObjectUrl("sourceObjectUrl");
  state.sourceObjectUrl = URL.createObjectURL(file);
  return loadImageFromUrl(state.sourceObjectUrl);
}

async function normalizeLargeSource(image, mimeType) {
  const totalPixels = image.width * image.height;

  if (totalPixels <= MAX_SOURCE_PIXELS) {
    return {
      image,
      width: image.width,
      height: image.height,
      objectUrl: state.sourceObjectUrl,
      wasDownscaled: false,
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

  const blob = await canvasToBlob(canvas, mimeType, 0.96);
  cleanupObjectUrl("sourceObjectUrl");
  state.sourceObjectUrl = URL.createObjectURL(blob);

  return {
    image: await loadImageFromUrl(state.sourceObjectUrl),
    width,
    height,
    objectUrl: state.sourceObjectUrl,
    wasDownscaled: true,
  };
}

function updateSourcePreview(normalized) {
  elements.originalPreview.src = normalized.objectUrl;
  elements.originalPreview.classList.add("visible");
  elements.originalEmpty.hidden = true;

  elements.compareBase.src = normalized.objectUrl;
  elements.compareBase.classList.add("visible");
  elements.modalCompareBase.src = normalized.objectUrl;
  elements.modalCompareBase.classList.add("visible");
}

function updateSourceMeta(file, normalized) {
  const details = [
    `${normalized.width} x ${normalized.height}`,
    formatFileSize(file.size),
    file.type.replace("image/", "").toUpperCase(),
  ];

  if (normalized.wasDownscaled) {
    details.push("optimized for performance");
  }

  elements.originalMeta.textContent = details.join(" - ");
}

async function processUpscale(requestedScale) {
  if (!state.sourceImage || !state.sourceFile || state.isProcessing) {
    return;
  }

  const jobId = ++state.activeJobId;
  state.isProcessing = true;
  setProcessingState(true);
  setBadge(elements.resultBadge, "Processing");
  updateStatus(`Enhancing image at ${requestedScale}x...`);
  updateProgress(0, "Preparing enhancement pipeline...");

  try {
    const options = getEnhancementOptions(requestedScale);
    const aiResult = await tryAiUpscale(requestedScale, options, jobId);

    let processedImageData;
    let resultWidth;
    let resultHeight;
    let appliedScale;

    if (aiResult) {
      processedImageData = aiResult.imageData;
      resultWidth = aiResult.width;
      resultHeight = aiResult.height;
      appliedScale = aiResult.appliedScale;
    } else {
      const scaleResult = await progressiveScaleImage(state.sourceImage, requestedScale, options, jobId);
      processedImageData = await processImageData(
        scaleResult.imageData,
        scaleResult.width,
        scaleResult.height,
        options,
        jobId
      );
      resultWidth = scaleResult.width;
      resultHeight = scaleResult.height;
      appliedScale = scaleResult.appliedScale;
    }

    if (jobId !== state.activeJobId) {
      return;
    }

    const resultBlob = await exportProcessedResult(processedImageData, resultWidth, resultHeight);

    cleanupObjectUrl("resultObjectUrl");
    state.resultBlob = resultBlob;
    state.resultMeta = {
      width: resultWidth,
      height: resultHeight,
      appliedScale,
    };
    state.resultObjectUrl = URL.createObjectURL(resultBlob);

    applyResultPreview();
    updateResultMeta();
    updateActionAvailability();
    updateOutputPreview();
    setBadge(elements.resultBadge, "Ready");
    updateProgress(100, "Enhancement complete.");

    const scaleLabel =
      Math.abs(appliedScale - requestedScale) < 0.05
        ? `${requestedScale}x`
        : `${appliedScale.toFixed(2)}x optimized`;

    updateStatus(`Upscale complete. Your ${scaleLabel} image is ready.`, "success");
    toast("Enhancement Complete", `Result is ready at ${scaleLabel}.`, "success");
  } catch (error) {
    if (error.message !== "Cancelled") {
      console.error(error);
      setBadge(elements.resultBadge, "Failed");
      updateStatus("Something went wrong while enhancing the image.", "error");
      toast("Enhancement Failed", "An unexpected error interrupted processing.", "error");
    }
  } finally {
    if (jobId === state.activeJobId) {
      state.isProcessing = false;
      setProcessingState(false);
    }
  }
}

function getEnhancementOptions(requestedScale) {
  const mode = elements.modeSelect.value;
  const modeConfig = MODE_CONFIG[mode];
  const denoiseStrength = Number(elements.denoiseSlider.value) / 100;
  const sharpnessStrength = Number(elements.sharpnessSlider.value) / 100;
  const outputScale = getSafeOutputScale(state.sourceImage.width, state.sourceImage.height, requestedScale);
  const aiProfile = elements.aiProfileSelect.value;
  const aiModel = AI_MODEL_CONFIG[aiProfile];

  return {
    mode,
    requestedScale,
    outputScale,
    denoiseStrength,
    sharpnessStrength,
    aiProfile,
    aiModel,
    ...modeConfig,
  };
}

async function tryAiUpscale(requestedScale, options, jobId) {
  if (!shouldUseAiPath(options)) {
    setAiStatus("Classic fallback");
    return null;
  }

  try {
    const sourceImageData = await getSourceImageData(state.sourceImage, jobId);
    const tileSize = getAiTileSize(options);

    updateProgress(6, "Loading AI model...");
    setAiStatus("Loading model");

    const response = await callAiWorker(
      {
        type: "upscale",
        payload: {
          modelUrl: options.aiModel.url,
          runtimeUrl: ORT_WEBGPU_CDN,
          requestedScale,
          modelScale: AI_MODEL_SCALE,
          aiProfile: options.aiProfile,
          width: sourceImageData.width,
          height: sourceImageData.height,
          tileSize,
          tileOverlap: AI_TILE_OVERLAP,
          preferGpu: Boolean(navigator.gpu),
          outputScale: options.outputScale,
          options: {
            denoiseStrength: options.denoiseStrength,
            sharpnessStrength: options.sharpnessStrength,
          },
          buffer: sourceImageData.data.buffer,
        },
      },
      [sourceImageData.data.buffer],
      jobId
    );

    setAiStatus(response.providerLabel || "AI ready");
    updateAiHint(response.providerLabel || "AI upscaling active.");

    return {
      imageData: new ImageData(new Uint8ClampedArray(response.buffer), response.width, response.height),
      width: response.width,
      height: response.height,
      appliedScale: response.appliedScale,
    };
  } catch (error) {
    console.error(error);
    setAiStatus("Fallback active");
    updateAiHint(
      "AI inference is unavailable or the model is missing, so Sat Upscaler switched to the classic enhancement path."
    );
    toast("AI Fallback", "Using the classic enhancement path for this device or model state.", "error");
    return null;
  }
}

function shouldUseAiPath(options) {
  if (!state.aiWorker) {
    return false;
  }

  const lowMemory = navigator.deviceMemory && navigator.deviceMemory <= 4;
  const sourcePixels = state.sourceImage.width * state.sourceImage.height;

  if (lowMemory && sourcePixels > 6_000_000) {
    return false;
  }

  return Boolean(options.aiModel?.url);
}

async function getSourceImageData(image, jobId) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { alpha: true, willReadFrequently: true });

  canvas.width = image.width;
  canvas.height = image.height;
  context.drawImage(image, 0, 0);

  await yieldToBrowser(jobId);

  return context.getImageData(0, 0, canvas.width, canvas.height);
}

function getAiTileSize(options) {
  const lowMemory = navigator.deviceMemory && navigator.deviceMemory <= 4;
  if (lowMemory) {
    return 128;
  }

  return options.aiModel.tileSize;
}

async function progressiveScaleImage(image, requestedScale, options, jobId) {
  const appliedScale = options.outputScale;
  const targetWidth = Math.max(1, Math.round(image.width * appliedScale));
  const targetHeight = Math.max(1, Math.round(image.height * appliedScale));
  const steps = getProgressiveScaleSteps(appliedScale);

  let currentCanvas = document.createElement("canvas");
  let currentContext = currentCanvas.getContext("2d", { alpha: true });
  currentCanvas.width = image.width;
  currentCanvas.height = image.height;
  currentContext.drawImage(image, 0, 0);

  updateProgress(4, "Analyzing source image...");
  await yieldToBrowser(jobId);

  let currentWidth = image.width;
  let currentHeight = image.height;

  for (let index = 0; index < steps.length; index += 1) {
    const stepFactor = steps[index];
    const nextWidth = Math.min(targetWidth, Math.max(1, Math.round(currentWidth * stepFactor)));
    const nextHeight = Math.min(targetHeight, Math.max(1, Math.round(currentHeight * stepFactor)));
    const nextCanvas = document.createElement("canvas");
    const nextContext = nextCanvas.getContext("2d", { alpha: true });

    nextCanvas.width = nextWidth;
    nextCanvas.height = nextHeight;
    nextContext.imageSmoothingEnabled = true;
    nextContext.imageSmoothingQuality = "high";
    nextContext.drawImage(currentCanvas, 0, 0, nextWidth, nextHeight);

    currentCanvas.width = 1;
    currentCanvas.height = 1;
    currentCanvas = nextCanvas;
    currentContext = nextContext;
    currentWidth = nextWidth;
    currentHeight = nextHeight;

    updateProgress(
      14 + Math.round(((index + 1) / steps.length) * 34),
      `Progressive upscale pass ${index + 1} of ${steps.length}...`
    );
    await yieldToBrowser(jobId);
  }

  const imageData = currentContext.getImageData(0, 0, currentWidth, currentHeight);
  return {
    imageData,
    width: currentWidth,
    height: currentHeight,
    appliedScale,
  };
}

async function processImageData(imageData, width, height, options, jobId) {
  updateProgress(56, "Running denoise and edge recovery...");
  await yieldToBrowser(jobId);

  const payload = {
    type: "process",
    payload: {
      width,
      height,
      buffer: imageData.data.buffer,
      options,
    },
  };

  if (state.worker) {
    const result = await callWorker(payload, [imageData.data.buffer], jobId);
    updateProgress(88, "Refining detail and preserving text...");
    await yieldToBrowser(jobId);
    return new ImageData(new Uint8ClampedArray(result.buffer), width, height);
  }

  const processed = processPixelsLocally(new Uint8ClampedArray(imageData.data), width, height, options);
  updateProgress(88, "Refining detail and preserving text...");
  await yieldToBrowser(jobId);
  return new ImageData(processed, width, height);
}

async function exportProcessedResult(imageData, width, height) {
  updateProgress(94, "Preparing export...");
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { alpha: true });
  const mimeType = elements.formatSelect.value;
  const quality = Number(elements.qualitySlider.value) / 100;

  canvas.width = width;
  canvas.height = height;
  context.putImageData(imageData, 0, 0);

  const blob = await canvasToBlob(canvas, mimeType, quality);
  canvas.width = 1;
  canvas.height = 1;
  return blob;
}

function applyResultPreview() {
  elements.upscaledPreview.src = state.resultObjectUrl;
  elements.upscaledPreview.classList.add("visible");
  elements.resultEmpty.hidden = true;

  elements.compareEnhanced.src = state.resultObjectUrl;
  elements.compareEnhanced.classList.add("visible");
  elements.compareOverlay.classList.add("visible");
  elements.compareSlider.disabled = false;
  elements.compareEmpty.hidden = true;

  elements.modalCompareEnhanced.src = state.resultObjectUrl;
  elements.modalCompareEnhanced.classList.add("visible");
  elements.modalCompareOverlay.classList.add("visible");

  elements.fullscreenBtn.disabled = false;
  updateZoom(elements.upscaledPreview, Number(elements.resultZoom.value || 100));
}

function updateResultMeta() {
  if (!state.resultMeta || !state.resultBlob) {
    return;
  }

  const formatLabel = elements.formatSelect.value.replace("image/", "").toUpperCase();
  const qualityValue = Number(elements.qualitySlider.value);
  const details = [
    `${state.resultMeta.width} x ${state.resultMeta.height}`,
    formatFileSize(state.resultBlob.size),
    formatLabel,
    `${qualityValue}% quality`,
  ];

  if (Math.abs(state.resultMeta.appliedScale - 2) > 0.05 && Math.abs(state.resultMeta.appliedScale - 4) > 0.05) {
    details.push("safe output limit");
  }

  elements.resultMeta.textContent = details.join(" - ");
}

function ensureWorker() {
  if (!("Worker" in window)) {
    return;
  }

  try {
    const workerSource = `
      self.onmessage = (event) => {
        const { type, id, payload } = event.data;
        if (type !== "process") {
          return;
        }

        const { width, height, buffer, options } = payload;
        const source = new Uint8ClampedArray(buffer);
        const processed = processPixels(source, width, height, options);
        self.postMessage({ id, buffer: processed.buffer }, [processed.buffer]);
      };

      function processPixels(source, width, height, options) {
        const base = new Uint8ClampedArray(source);
        const denoised = denoise(base, width, height, options);
        const blurred = boxBlur(denoised, width, height);
        const edges = edgeMap(denoised, width, height);
        const result = new Uint8ClampedArray(base.length);

        for (let index = 0; index < base.length; index += 4) {
          const edgeStrength = edges[index] / 255;
          const localDetail = (denoised[index] - blurred[index]) / 255;
          const textMask = edgeStrength > 0.24 ? 1 : 0;

          for (let channel = 0; channel < 3; channel += 1) {
            const original = denoised[index + channel];
            const blurValue = blurred[index + channel];
            const detail = original - blurValue;
            const edgeBoost = detail * options.edgeBoost * edgeStrength * (1 + options.sharpnessStrength);
            const detailBoost = detail * options.detailBoost * (1 + localDetail);
            const textBoost = detail * options.textBoost * textMask;
            const saturationOffset = (original - 128) * options.saturationBoost * 0.08;
            result[index + channel] = clamp(original + edgeBoost + detailBoost + textBoost + saturationOffset);
          }

          result[index + 3] = denoised[index + 3];
        }

        return result;
      }

      function denoise(source, width, height, options) {
        const blurred = boxBlur(source, width, height);
        const output = new Uint8ClampedArray(source.length);
        const mix = Math.min(0.42, options.denoiseBias + options.denoiseStrength * 0.34);

        for (let index = 0; index < source.length; index += 4) {
          for (let channel = 0; channel < 3; channel += 1) {
            output[index + channel] = clamp(source[index + channel] * (1 - mix) + blurred[index + channel] * mix);
          }

          output[index + 3] = source[index + 3];
        }

        return output;
      }

      function edgeMap(source, width, height) {
        const output = new Uint8ClampedArray(source.length);
        for (let y = 1; y < height - 1; y += 1) {
          for (let x = 1; x < width - 1; x += 1) {
            const i = (y * width + x) * 4;
            const left = intensity(source, ((y * width) + (x - 1)) * 4);
            const right = intensity(source, ((y * width) + (x + 1)) * 4);
            const top = intensity(source, (((y - 1) * width) + x) * 4);
            const bottom = intensity(source, (((y + 1) * width) + x) * 4);
            const value = Math.min(255, Math.abs(right - left) + Math.abs(bottom - top));
            output[i] = value;
            output[i + 1] = value;
            output[i + 2] = value;
            output[i + 3] = source[i + 3];
          }
        }
        return output;
      }

      function boxBlur(source, width, height) {
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

      function intensity(source, index) {
        return source[index] * 0.299 + source[index + 1] * 0.587 + source[index + 2] * 0.114;
      }

      function clamp(value) {
        return Math.max(0, Math.min(255, Math.round(value)));
      }

      function clampIndex(value, min, max) {
        return Math.max(min, Math.min(max, value));
      }
    `;

    const blob = new Blob([workerSource], { type: "application/javascript" });
    const workerUrl = URL.createObjectURL(blob);
    state.worker = new Worker(workerUrl);
    URL.revokeObjectURL(workerUrl);

    state.worker.addEventListener("message", (event) => {
      const { id, buffer } = event.data;
      const resolver = state.pendingWorkerResolvers.get(id);
      if (!resolver) {
        return;
      }

      state.pendingWorkerResolvers.delete(id);
      resolver.resolve({ buffer });
    });

    state.worker.addEventListener("error", () => {
      state.worker?.terminate();
      state.worker = null;
    });
  } catch (error) {
    console.error(error);
    state.worker = null;
  }
}

function ensureAiWorker() {
  if (!("Worker" in window)) {
    setAiStatus("Worker unavailable");
    updateAiHint("This browser does not support worker-based AI inference, so classic upscale will be used.");
    return;
  }

  try {
    state.aiWorker = new Worker("./ai-upscale-worker.js");

    state.aiWorker.addEventListener("message", (event) => {
      const { id, type } = event.data;

      if (type === "progress") {
        handleAiProgress(event.data);
        return;
      }

      if (type === "error") {
        const resolver = state.pendingAiResolvers.get(id);
        if (!resolver) {
          return;
        }

        state.pendingAiResolvers.delete(id);
        resolver.reject(new Error(event.data.message || "AI worker error"));
        return;
      }

      const resolver = state.pendingAiResolvers.get(id);
      if (!resolver) {
        return;
      }

      state.pendingAiResolvers.delete(id);
      resolver.resolve(event.data);
    });

    state.aiWorker.addEventListener("error", () => {
      state.aiWorker?.terminate();
      state.aiWorker = null;
      setAiStatus("Worker failed");
    });
  } catch (error) {
    console.error(error);
    state.aiWorker = null;
    setAiStatus("Worker failed");
  }
}

function handleAiProgress(message) {
  switch (message.phase) {
    case "runtime-loading":
      updateProgress(8, "Loading ONNX runtime...");
      setAiStatus("Loading runtime");
      break;
    case "model-download":
      if (message.total > 0) {
        const percent = 8 + Math.round((message.loaded / message.total) * 18);
        updateProgress(percent, "Downloading AI model...");
      } else {
        updateProgress(16, "Downloading AI model...");
      }
      setAiStatus("Downloading model");
      break;
    case "session-creating":
      updateProgress(30, "Creating AI inference session...");
      setAiStatus("Starting session");
      break;
    case "session-ready":
      updateProgress(38, `AI session ready (${message.providerLabel || "wasm"})`);
      setAiStatus(message.providerLabel || "AI ready");
      break;
    case "tiling":
      updateProgress(44, "Preparing tiled inference...");
      setAiStatus("Preparing tiles");
      break;
    case "tile":
      updateProgress(
        44 + Math.round((message.completed / message.total) * 48),
        `Running AI tile ${message.completed} of ${message.total}...`
      );
      setAiStatus("Inferencing");
      break;
    case "downscale":
      updateProgress(94, "Finalizing scaled output...");
      setAiStatus("Finishing");
      break;
    default:
      break;
  }
}

function callWorker(message, transfer, jobId) {
  return new Promise((resolve, reject) => {
    if (!state.worker) {
      reject(new Error("Worker unavailable"));
      return;
    }

    const id = `${jobId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    state.pendingWorkerResolvers.set(id, { resolve, reject });
    state.worker.postMessage({ ...message, id }, transfer);
  });
}

function callAiWorker(message, transfer, jobId) {
  return new Promise((resolve, reject) => {
    if (!state.aiWorker) {
      reject(new Error("AI worker unavailable"));
      return;
    }

    const id = `ai-${jobId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    state.pendingAiResolvers.set(id, { resolve, reject });
    state.aiWorker.postMessage({ ...message, id }, transfer);
  });
}

function processPixelsLocally(source, width, height, options) {
  const denoised = localDenoise(source, width, height, options);
  const blurred = localBoxBlur(denoised, width, height);
  const edges = localEdgeMap(denoised, width, height);
  const result = new Uint8ClampedArray(source.length);

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
      result[index + channel] = clampColor(original + edgeBoost + detailBoost + textBoost + saturationOffset);
    }

    result[index + 3] = denoised[index + 3];
  }

  return result;
}

function localDenoise(source, width, height, options) {
  const blurred = localBoxBlur(source, width, height);
  const output = new Uint8ClampedArray(source.length);
  const mix = Math.min(0.42, options.denoiseBias + options.denoiseStrength * 0.34);

  for (let index = 0; index < source.length; index += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      output[index + channel] = clampColor(source[index + channel] * (1 - mix) + blurred[index + channel] * mix);
    }

    output[index + 3] = source[index + 3];
  }

  return output;
}

function localBoxBlur(source, width, height) {
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
          const px = clamp(x + ox, 0, width - 1);
          const py = clamp(y + oy, 0, height - 1);
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

function localEdgeMap(source, width, height) {
  const output = new Uint8ClampedArray(source.length);

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = (y * width + x) * 4;
      const left = localIntensity(source, ((y * width) + (x - 1)) * 4);
      const right = localIntensity(source, ((y * width) + (x + 1)) * 4);
      const top = localIntensity(source, (((y - 1) * width) + x) * 4);
      const bottom = localIntensity(source, (((y + 1) * width) + x) * 4);
      const value = Math.min(255, Math.abs(right - left) + Math.abs(bottom - top));
      output[index] = value;
      output[index + 1] = value;
      output[index + 2] = value;
      output[index + 3] = source[index + 3];
    }
  }

  return output;
}

function localIntensity(source, index) {
  return source[index] * 0.299 + source[index + 1] * 0.587 + source[index + 2] * 0.114;
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

function updateActionAvailability() {
  const hasSource = Boolean(state.sourceImage);
  const hasResult = Boolean(state.resultObjectUrl);

  elements.upscale2xBtn.disabled = state.isProcessing || !hasSource;
  elements.upscale4xBtn.disabled = state.isProcessing || !hasSource;
  elements.downloadBtn.disabled = state.isProcessing || !hasResult;
  elements.fullscreenBtn.disabled = !hasResult;
}

function setProcessingState(isProcessing) {
  state.isProcessing = isProcessing;
  elements.loader.hidden = !isProcessing;
  elements.fileInput.disabled = isProcessing;
  elements.modeSelect.disabled = isProcessing;
  elements.aiProfileSelect.disabled = isProcessing;
  elements.denoiseSlider.disabled = isProcessing;
  elements.sharpnessSlider.disabled = isProcessing;
  elements.formatSelect.disabled = isProcessing;
  elements.qualitySlider.disabled = isProcessing;
  elements.filenameInput.disabled = isProcessing;
  elements.resetBtn.disabled = isProcessing;
  updateActionAvailability();
}

function updateComparison(value) {
  state.compareValue = clamp(value, 0, 100);
  elements.compareSlider.value = String(state.compareValue);
  elements.compareOverlay.style.width = `${state.compareValue}%`;
  elements.compareLine.style.left = `${state.compareValue}%`;
}

function updateModalComparison(value) {
  elements.modalCompareSlider.value = String(value);
  elements.modalCompareOverlay.style.width = `${value}%`;
  elements.modalCompareLine.style.left = `${value}%`;
}

function updateProgress(percent, stage) {
  const safePercent = clamp(Math.round(percent), 0, 100);
  const text = `${safePercent}%`;

  elements.progressValue.textContent = text;
  elements.progressStage.textContent = stage;
  elements.progressText.textContent = text;
  elements.progressBarFill.style.width = text;
}

function updateStatus(message, type = "") {
  elements.statusMessage.textContent = message;
  elements.statusMessage.className = "status-message";
  if (type) {
    elements.statusMessage.classList.add(type);
  }
}

function updateEstimate() {
  if (!state.sourceImage) {
    elements.timeEstimate.textContent = "Estimate: waiting";
    return;
  }

  const megapixels = (state.sourceImage.width * state.sourceImage.height) / 1_000_000;
  const modeFactor = {
    balanced: 1,
    sharp: 1.18,
    photo: 1.08,
    anime: 1.14,
  }[elements.modeSelect.value];
  const aiFactor = elements.aiProfileSelect.value === "anime" ? 1.18 : 1.08;
  const denoiseFactor = 1 + Number(elements.denoiseSlider.value) / 220;
  const sharpFactor = 1 + Number(elements.sharpnessSlider.value) / 260;
  const estimateSeconds = Math.max(
    1,
    Math.round(megapixels * modeFactor * aiFactor * denoiseFactor * sharpFactor * 1.8)
  );

  elements.timeEstimate.textContent = `Estimate: ~${estimateSeconds}s`;
}

function updateOutputPreview() {
  if (!state.sourceImage) {
    elements.outputPreview.textContent = "Output: waiting";
    return;
  }

  const scale = getSafeOutputScale(state.sourceImage.width, state.sourceImage.height, 4);
  const width = Math.round(state.sourceImage.width * scale);
  const height = Math.round(state.sourceImage.height * scale);
  const format = elements.formatSelect.value.replace("image/", "").toUpperCase();
  elements.outputPreview.textContent = `Output: up to ${width} x ${height} ${format}`;
}

function syncControlLabels() {
  elements.denoiseValue.textContent = `${elements.denoiseSlider.value}%`;
  elements.sharpnessValue.textContent = `${elements.sharpnessSlider.value}%`;
  elements.qualityValue.textContent = `${elements.qualitySlider.value}%`;
}

function updateAiHint(message) {
  if (message) {
    elements.aiHint.textContent = message;
    return;
  }

  const profile = AI_MODEL_CONFIG[elements.aiProfileSelect.value];
  elements.aiHint.textContent = `AI profile: ${profile.label}. The ONNX model is lazy-loaded only when you start an upscale.`;
}

function setAiStatus(text) {
  elements.aiStatus.textContent = text;
}

function updateZoom(imageElement, zoom) {
  if (!(imageElement instanceof HTMLImageElement)) {
    return;
  }

  imageElement.style.transform = `scale(${zoom / 100})`;
}

function openFullscreenModal() {
  if (!state.resultObjectUrl) {
    return;
  }

  elements.fullscreenModal.hidden = false;
  document.body.style.overflow = "hidden";
  updateModalComparison(state.compareValue);
}

function closeFullscreenModal() {
  if (elements.fullscreenModal.hidden) {
    return;
  }

  elements.fullscreenModal.hidden = true;
  document.body.style.overflow = "";
}

function downloadResult() {
  if (!state.resultObjectUrl || !state.resultBlob) {
    toast("Nothing To Download", "Run an upscale before exporting.", "error");
    return;
  }

  const format = elements.formatSelect.value;
  const extension = getExtensionFromType(format);
  const filename = sanitizeFilename(elements.filenameInput.value.trim() || "sat-upscaled");
  const anchor = document.createElement("a");

  anchor.href = state.resultObjectUrl;
  anchor.download = `${filename}.${extension}`;
  anchor.click();
  toast("Download Started", `Exporting ${filename}.${extension}`, "success");
}

function resetWorkspace() {
  state.activeJobId += 1;
  state.isProcessing = false;
  state.sourceFile = null;
  state.sourceImage = null;
  state.resultBlob = null;
  state.resultMeta = null;

  cleanupObjectUrl("sourceObjectUrl");
  cleanupObjectUrl("resultObjectUrl");

  elements.fileInput.value = "";
  elements.filenameInput.value = "sat-upscaled";
  elements.originalPreview.removeAttribute("src");
  elements.upscaledPreview.removeAttribute("src");
  elements.compareBase.removeAttribute("src");
  elements.compareEnhanced.removeAttribute("src");
  elements.modalCompareBase.removeAttribute("src");
  elements.modalCompareEnhanced.removeAttribute("src");

  elements.originalPreview.classList.remove("visible");
  elements.upscaledPreview.classList.remove("visible");
  elements.compareBase.classList.remove("visible");
  elements.compareEnhanced.classList.remove("visible");
  elements.compareOverlay.classList.remove("visible");
  elements.modalCompareBase.classList.remove("visible");
  elements.modalCompareEnhanced.classList.remove("visible");
  elements.modalCompareOverlay.classList.remove("visible");

  elements.originalEmpty.hidden = false;
  elements.resultEmpty.hidden = false;
  elements.compareEmpty.hidden = false;
  elements.compareSlider.disabled = true;
  elements.fullscreenBtn.disabled = true;

  elements.originalMeta.textContent = "No image loaded";
  elements.resultMeta.textContent = "Waiting to process";
  setBadge(elements.originalBadge, "Idle");
  setBadge(elements.resultBadge, "Waiting");
  updateComparison(DEFAULT_COMPARE_VALUE);
  updateModalComparison(DEFAULT_COMPARE_VALUE);
  updateProgress(0, "Preparing enhancement pipeline...");
  updateStatus("Workspace reset. Upload a new image to continue.");
  setAiStatus("Lazy-loaded");
  updateAiHint();
  updateEstimate();
  updateOutputPreview();
  closeFullscreenModal();
  setProcessingState(false);
}

function clearResultState() {
  cleanupObjectUrl("resultObjectUrl");
  state.resultBlob = null;
  state.resultMeta = null;

  elements.upscaledPreview.removeAttribute("src");
  elements.compareEnhanced.removeAttribute("src");
  elements.modalCompareEnhanced.removeAttribute("src");
  elements.upscaledPreview.classList.remove("visible");
  elements.compareEnhanced.classList.remove("visible");
  elements.compareOverlay.classList.remove("visible");
  elements.modalCompareEnhanced.classList.remove("visible");
  elements.modalCompareOverlay.classList.remove("visible");
  elements.resultEmpty.hidden = false;
  elements.compareEmpty.hidden = false;
  elements.compareSlider.disabled = true;
  elements.fullscreenBtn.disabled = true;
  elements.resultMeta.textContent = "Waiting to process";
  setBadge(elements.resultBadge, "Waiting");
  updateProgress(0, "Preparing enhancement pipeline...");
  setAiStatus("Lazy-loaded");
  updateAiHint();
}

function pulseUploadAnimation() {
  elements.dropzone.classList.remove("uploaded");
  elements.uploadBurst.classList.remove("active");
  requestAnimationFrame(() => {
    elements.dropzone.classList.add("uploaded");
    elements.uploadBurst.classList.add("active");
  });

  setTimeout(() => {
    elements.dropzone.classList.remove("uploaded");
    elements.uploadBurst.classList.remove("active");
  }, 600);
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

function cleanupObjectUrl(key) {
  if (state[key]) {
    URL.revokeObjectURL(state[key]);
    state[key] = "";
  }
}

function getFileFromClipboard(clipboardData) {
  if (!clipboardData?.items) {
    return null;
  }

  for (const item of clipboardData.items) {
    if (item.kind === "file" && SUPPORTED_TYPES.includes(item.type)) {
      return item.getAsFile();
    }
  }

  return null;
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

async function yieldToBrowser(jobId) {
  if (jobId !== state.activeJobId) {
    throw new Error("Cancelled");
  }

  await new Promise((resolve) => {
    requestAnimationFrame(() => setTimeout(resolve, 0));
  });
}

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

function sanitizeFilename(name) {
  return name.replace(/[<>:"/\\|?*\x00-\x1F]/g, "-").replace(/\s+/g, "-");
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function setBadge(element, text) {
  element.textContent = text;
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clampColor(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function isTypingTarget(target) {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  );
}
