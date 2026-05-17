const sessionCache = new Map();
let ortReady = false;

self.onmessage = async (event) => {
  const { id, type, payload, itemId } = event.data;

  if (type !== "upscale") {
    return;
  }

  try {
    const result = await runUpscale(payload, id, itemId);
    self.postMessage({ id, itemId, ...result }, [result.buffer]);
  } catch (error) {
    self.postMessage({
      id,
      itemId,
      type: "error",
      message: error instanceof Error ? error.message : "AI upscale failed",
    });
  }
};

async function runUpscale(payload, id, itemId) {
  await ensureOrtRuntime(payload.runtimeUrl, id, itemId);

  const sessionData = await getSession(payload.modelUrl, payload.preferGpu, id, itemId);
  const sourcePixels = new Uint8ClampedArray(payload.buffer);
  const sourceImage = new ImageData(sourcePixels, payload.width, payload.height);
  const sourceCanvas = new OffscreenCanvas(payload.width, payload.height);
  const sourceContext = sourceCanvas.getContext("2d", { alpha: true, willReadFrequently: true });
  sourceContext.putImageData(sourceImage, 0, 0);

  const upscaledWidth = payload.width * payload.modelScale;
  const upscaledHeight = payload.height * payload.modelScale;
  const resultCanvas = new OffscreenCanvas(upscaledWidth, upscaledHeight);
  const resultContext = resultCanvas.getContext("2d", { alpha: true });
  const tilesX = Math.ceil(payload.width / payload.tileSize);
  const tilesY = Math.ceil(payload.height / payload.tileSize);
  const totalTiles = tilesX * tilesY;

  postProgress(id, itemId, "tiling");

  let tileIndex = 0;

  for (let tileY = 0; tileY < tilesY; tileY += 1) {
    for (let tileX = 0; tileX < tilesX; tileX += 1) {
      tileIndex += 1;

      const startX = tileX * payload.tileSize;
      const startY = tileY * payload.tileSize;
      const endX = Math.min(payload.width, startX + payload.tileSize);
      const endY = Math.min(payload.height, startY + payload.tileSize);

      const padLeft = Math.min(payload.tileOverlap, startX);
      const padTop = Math.min(payload.tileOverlap, startY);
      const padRight = Math.min(payload.tileOverlap, payload.width - endX);
      const padBottom = Math.min(payload.tileOverlap, payload.height - endY);

      const readX = startX - padLeft;
      const readY = startY - padTop;
      const readWidth = endX - startX + padLeft + padRight;
      const readHeight = endY - startY + padTop + padBottom;

      const tilePixels = sourceContext.getImageData(readX, readY, readWidth, readHeight);
      const feeds = {};
      feeds[sessionData.inputName] = createInputTensor(tilePixels, readWidth, readHeight);

      const outputs = await sessionData.session.run(feeds);
      const outputTensor = outputs[sessionData.outputName];
      const tileImageData = tensorToImageData(outputTensor, readWidth * payload.modelScale, readHeight * payload.modelScale);

      const cropX = padLeft * payload.modelScale;
      const cropY = padTop * payload.modelScale;
      const cropWidth = (endX - startX) * payload.modelScale;
      const cropHeight = (endY - startY) * payload.modelScale;
      const drawX = startX * payload.modelScale;
      const drawY = startY * payload.modelScale;

      resultContext.putImageData(tileImageData, drawX - cropX, drawY - cropY, cropX, cropY, cropWidth, cropHeight);

      postProgress(id, itemId, "tile", {
        completed: tileIndex,
        total: totalTiles,
      });
    }
  }

  const targetWidth = Math.max(1, Math.round(payload.width * payload.outputScale));
  const targetHeight = Math.max(1, Math.round(payload.height * payload.outputScale));

  if (targetWidth !== upscaledWidth || targetHeight !== upscaledHeight) {
    postProgress(id, itemId, "downscale");
    const finalCanvas = new OffscreenCanvas(targetWidth, targetHeight);
    const finalContext = finalCanvas.getContext("2d", { alpha: true });
    finalContext.imageSmoothingEnabled = true;
    finalContext.imageSmoothingQuality = "high";
    finalContext.drawImage(resultCanvas, 0, 0, targetWidth, targetHeight);
    const finalData = finalContext.getImageData(0, 0, targetWidth, targetHeight);

    return {
      type: "result",
      buffer: finalData.data.buffer,
      width: targetWidth,
      height: targetHeight,
      appliedScale: payload.outputScale,
      providerLabel: sessionData.providerLabel,
    };
  }

  const resultData = resultContext.getImageData(0, 0, upscaledWidth, upscaledHeight);
  return {
    type: "result",
    buffer: resultData.data.buffer,
    width: upscaledWidth,
    height: upscaledHeight,
    appliedScale: payload.outputScale,
    providerLabel: sessionData.providerLabel,
  };
}

async function ensureOrtRuntime(runtimeUrl, id, itemId) {
  if (ortReady) {
    return;
  }

  postProgress(id, itemId, "runtime-loading");
  importScripts(runtimeUrl);
  self.ort.env.wasm.numThreads = Math.max(1, Math.min(4, Math.floor((self.navigator.hardwareConcurrency || 4) / 2)));
  self.ort.env.wasm.proxy = false;
  ortReady = true;
}

async function getSession(modelUrl, preferGpu, id, itemId) {
  const cacheKey = `${modelUrl}::${preferGpu ? "gpu" : "cpu"}`;

  if (sessionCache.has(cacheKey)) {
    return sessionCache.get(cacheKey);
  }

  const modelBuffer = await fetchModelWithProgress(modelUrl, id, itemId);
  postProgress(id, itemId, "session-creating");

  const executionProviders = preferGpu ? ["webgpu", "wasm"] : ["wasm"];
  const session = await self.ort.InferenceSession.create(modelBuffer, {
    executionProviders,
    graphOptimizationLevel: "all",
  });

  const sessionData = {
    session,
    inputName: session.inputNames[0],
    outputName: session.outputNames[0],
    providerLabel: preferGpu ? "WebGPU AI" : "WASM AI",
  };

  sessionCache.set(cacheKey, sessionData);
  postProgress(id, itemId, "session-ready", { providerLabel: sessionData.providerLabel });
  return sessionData;
}

async function fetchModelWithProgress(modelUrl, id, itemId) {
  const response = await fetch(modelUrl);
  if (!response.ok) {
    throw new Error(`Model request failed: ${response.status}`);
  }

  const contentLength = Number(response.headers.get("content-length") || 0);
  if (!response.body) {
    return response.arrayBuffer();
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    chunks.push(value);
    loaded += value.byteLength;
    postProgress(id, itemId, "model-download", {
      loaded,
      total: contentLength,
    });
  }

  const modelBuffer = new Uint8Array(loaded);
  let offset = 0;

  for (const chunk of chunks) {
    modelBuffer.set(chunk, offset);
    offset += chunk.byteLength;
  }

  return modelBuffer.buffer;
}

function createInputTensor(imageData, width, height) {
  const input = new Float32Array(1 * 3 * height * width);
  const { data } = imageData;
  const planeSize = width * height;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pixelIndex = (y * width + x) * 4;
      const offset = y * width + x;
      input[offset] = data[pixelIndex] / 255;
      input[planeSize + offset] = data[pixelIndex + 1] / 255;
      input[planeSize * 2 + offset] = data[pixelIndex + 2] / 255;
    }
  }

  return new self.ort.Tensor("float32", input, [1, 3, height, width]);
}

function tensorToImageData(tensor, width, height) {
  const output = new Uint8ClampedArray(width * height * 4);
  const [batch, channels] = tensor.dims;
  if (batch !== 1 || channels !== 3) {
    throw new Error("Unexpected model output shape.");
  }

  const planeSize = width * height;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = y * width + x;
      const pixelIndex = offset * 4;
      output[pixelIndex] = clampColor(tensor.data[offset] * 255);
      output[pixelIndex + 1] = clampColor(tensor.data[planeSize + offset] * 255);
      output[pixelIndex + 2] = clampColor(tensor.data[planeSize * 2 + offset] * 255);
      output[pixelIndex + 3] = 255;
    }
  }

  return new ImageData(output, width, height);
}

function postProgress(id, itemId, phase, extra = {}) {
  self.postMessage({
    id,
    itemId,
    type: "progress",
    phase,
    ...extra,
  });
}

function clampColor(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}
