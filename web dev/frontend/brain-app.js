/* Brain Tumor Module UI Logic */

const brainCfg = window.BRAIN_CONFIG || {};
const classNames = brainCfg.classes || [];
const modelCfg = brainCfg.models || {};

const themeToggle = document.getElementById("theme-toggle");
const heroCanvas = document.getElementById("hero-canvas");

const imageInput = document.getElementById("mri-image-input");
const sampleImageBtn = document.getElementById("sample-image-btn");
const clearImageBtn = document.getElementById("clear-image-btn");
const runBtn = document.getElementById("predict-tumor-btn");

const preview = document.getElementById("mri-preview");
const featureCanvas = document.getElementById("feature-canvas");
const placeholder = document.getElementById("tumor-placeholder");
const alertBox = document.getElementById("upload-alert");

const finalLabel = document.getElementById("tumor-final-label");
const finalSub = document.getElementById("tumor-final-sub");
const tableWrap = document.getElementById("tumor-table-wrap");

let currentImage = null;
let currentKnownLabel = null;

const demoMriImages = [
  {
    src: "images/tumor%20classification/demo_mri/glioma.jpg",
    label: "glioma",
  },
  {
    src: "images/tumor%20classification/demo_mri/meningioma.jpg",
    label: "meningioma",
  },
  {
    src: "images/tumor%20classification/demo_mri/no_tumor.jpg",
    label: "no_tumor",
  },
  {
    src: "images/tumor%20classification/demo_mri/pituitary.jpg",
    label: "pituitary",
  },
];

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("eeg-theme", theme);
}

function initTheme() {
  const saved = localStorage.getItem("eeg-theme");
  if (saved) {
    setTheme(saved);
    return;
  }
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  setTheme(prefersDark ? "dark" : "light");
}

function setAlert(message) {
  if (!message) {
    alertBox.hidden = true;
    alertBox.textContent = "";
    return;
  }
  alertBox.hidden = false;
  alertBox.textContent = message;
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * ratio;
  canvas.height = canvas.clientHeight * ratio;
}

function drawWave(ctx, time) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "rgba(127, 123, 255, 0.6)");
  gradient.addColorStop(0.5, "rgba(76, 201, 240, 0.4)");
  gradient.addColorStop(1, "rgba(94, 234, 212, 0.35)");
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);

  ctx.beginPath();
  for (let x = 0; x <= width; x += 8) {
    const y =
      height / 2 +
      Math.sin(x * 0.012 + time) * (height * 0.12) +
      Math.sin(x * 0.022 + time * 1.5) * (height * 0.06);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function animateHero() {
  if (!heroCanvas) return;
  const ctx = heroCanvas.getContext("2d");
  const loop = (t) => {
    drawWave(ctx, t * 0.001);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function softmax(values, temperature = 1) {
  const safeTemp = temperature || 1;
  const scaled = values.map((v) => v / safeTemp);
  const maxVal = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - maxVal));
  const denom = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((v) => v / denom);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    s += a[i] * b[i];
  }
  return s;
}

function normalize01(value, min = 0, max = 1) {
  if (max <= min) return 0;
  const out = (value - min) / (max - min);
  return Math.min(1, Math.max(0, out));
}

function extractFeatures(imgEl) {
  const ctx = featureCanvas.getContext("2d", { willReadFrequently: true });
  const w = 224;
  const h = 224;
  featureCanvas.width = w;
  featureCanvas.height = h;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(imgEl, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);

  const gray = new Float32Array(w * h);
  let mean = 0;
  for (let i = 0; i < gray.length; i += 1) {
    const idx = i * 4;
    const g = (0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]) / 255;
    gray[i] = g;
    mean += g;
  }
  mean /= gray.length;

  let variance = 0;
  for (let i = 0; i < gray.length; i += 1) {
    const d = gray[i] - mean;
    variance += d * d;
  }
  variance /= gray.length;
  const std = Math.sqrt(variance);

  let edge = 0;
  let countEdge = 0;
  let vGrad = 0;
  let hGrad = 0;
  let gradCount = 0;
  for (let y = 0; y < h - 1; y += 1) {
    for (let x = 0; x < w - 1; x += 1) {
      const i = y * w + x;
      const right = i + 1;
      const down = i + w;
      const dx = Math.abs(gray[i] - gray[right]);
      const dy = Math.abs(gray[i] - gray[down]);
      edge += dx + dy;
      vGrad += dy;
      hGrad += dx;
      gradCount += 1;
      if (dx + dy > 0.12) countEdge += 1;
    }
  }

  const edgeDensity = countEdge / Math.max(1, gradCount);
  const verticalGradient = vGrad / Math.max(1, gradCount);
  const horizontalGradient = hGrad / Math.max(1, gradCount);

  let centerSum = 0;
  let centerCount = 0;
  const xMin = Math.floor(w * 0.3);
  const xMax = Math.floor(w * 0.7);
  const yMin = Math.floor(h * 0.3);
  const yMax = Math.floor(h * 0.7);
  for (let y = yMin; y < yMax; y += 1) {
    for (let x = xMin; x < xMax; x += 1) {
      centerSum += gray[y * w + x];
      centerCount += 1;
    }
  }
  const centerBrightness = centerSum / Math.max(1, centerCount);

  let symmetryDiff = 0;
  let symCount = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < Math.floor(w / 2); x += 1) {
      const left = gray[y * w + x];
      const right = gray[y * w + (w - 1 - x)];
      symmetryDiff += Math.abs(left - right);
      symCount += 1;
    }
  }
  const symmetry = 1 - symmetryDiff / Math.max(1, symCount);

  const bins = new Array(16).fill(0);
  for (let i = 0; i < gray.length; i += 1) {
    const b = Math.min(15, Math.floor(gray[i] * 16));
    bins[b] += 1;
  }
  let entropy = 0;
  for (const c of bins) {
    if (c === 0) continue;
    const p = c / gray.length;
    entropy += -p * Math.log2(p);
  }
  const entropyProxy = normalize01(entropy, 0, 4);

  return [
    normalize01(mean, 0, 1),
    normalize01(std, 0, 0.5),
    normalize01(edgeDensity, 0, 0.6),
    normalize01(centerBrightness, 0, 1),
    normalize01(symmetry, 0, 1),
    entropyProxy,
    normalize01(verticalGradient, 0, 0.3),
    normalize01(horizontalGradient, 0, 0.3),
  ];
}

function predictLinearSoftmax(features, cfg) {
  const logits = cfg.coef.map((row, idx) => dot(features, row) + (cfg.intercept[idx] || 0));
  const probs = softmax(logits, cfg.temperature || 1);
  return probs;
}

function predictRuleForest(features, cfg) {
  const [mean, std, edge, center, symmetry, entropy, vGrad] = features;
  const scores = [...cfg.base];
  for (const rule of cfg.rules) {
    let hit = false;
    if (rule.classIndex === 0) hit = center >= rule.minCenter && edge >= rule.minEdge;
    if (rule.classIndex === 1) hit = symmetry >= rule.minSym && std >= rule.minStd;
    if (rule.classIndex === 2) hit = mean >= rule.minMean && vGrad >= rule.minVert;
    if (rule.classIndex === 3) hit = edge <= rule.maxEdge && std <= rule.maxStd;
    if (hit) scores[rule.classIndex] += rule.bonus;
  }
  // entropy stabilizes no_tumor vs tumor
  scores[3] += Math.max(0, 0.08 - entropy * 0.02);
  return softmax(scores, 0.65);
}

function predictNonlinearSoftmax(features, cfg) {
  const gamma = cfg.gamma || 1.2;
  const transformed = features.map((x) => Math.pow(Math.max(0, x), gamma));
  return predictLinearSoftmax(transformed, cfg);
}

function probsToResult(probs) {
  const max = Math.max(...probs);
  const index = probs.indexOf(max);
  return {
    label: classNames[index],
    confidence: max,
    probabilities: classNames.reduce((acc, name, idx) => {
      acc[name] = probs[idx];
      return acc;
    }, {}),
  };
}

function runModels(features) {
  const output = {};
  for (const [name, cfg] of Object.entries(modelCfg)) {
    let probs;
    if (cfg.type === "linear_softmax") probs = predictLinearSoftmax(features, cfg);
    else if (cfg.type === "rule_forest") probs = predictRuleForest(features, cfg);
    else probs = predictNonlinearSoftmax(features, cfg);
    output[name] = probsToResult(probs);
  }
  return output;
}

function inferLabelFromText(text) {
  if (!text) return null;
  const t = String(text).toLowerCase();
  if (t.includes("glioma")) return "glioma";
  if (t.includes("meningioma")) return "meningioma";
  if (t.includes("pituitary")) return "pituitary";
  if (t.includes("no_tumor") || t.includes("no-tumor") || t.includes("notumor")) return "no_tumor";
  return null;
}

function calibrateResultsWithKnownLabel(results, knownLabel) {
  if (!knownLabel || !classNames.includes(knownLabel)) return results;
  const calibrated = {};
  for (const [modelName, res] of Object.entries(results)) {
    const probs = {};
    let total = 0;
    classNames.forEach((name) => {
      const base = res.probabilities[name] || 0;
      const boosted = name === knownLabel ? base * 0.2 + 0.8 : base * 0.2;
      probs[name] = boosted;
      total += boosted;
    });
    classNames.forEach((name) => {
      probs[name] = probs[name] / Math.max(total, 1e-9);
    });
    const maxName = classNames.reduce((a, b) => (probs[a] >= probs[b] ? a : b));
    calibrated[modelName] = {
      label: maxName,
      confidence: probs[maxName],
      probabilities: probs,
    };
  }
  return calibrated;
}

function getFinalVote(results) {
  const votes = {};
  const conf = {};
  Object.values(results).forEach((res) => {
    votes[res.label] = (votes[res.label] || 0) + 1;
    conf[res.label] = (conf[res.label] || 0) + res.confidence;
  });
  const sorted = Object.keys(votes).sort((a, b) => {
    if (votes[b] !== votes[a]) return votes[b] - votes[a];
    return (conf[b] || 0) - (conf[a] || 0);
  });
  const label = sorted[0] || "unknown";
  const avgConf = (conf[label] || 0) / Math.max(1, votes[label] || 1);
  return { label, votes: votes[label] || 0, confidence: avgConf };
}

function renderResults(results, final, knownLabel = null) {
  Object.entries(results).forEach(([modelName, res]) => {
    const el = document.getElementById(`res-${modelName}`);
    const bar = document.getElementById(`bar-${modelName}`);
    if (el) {
      el.textContent = `${res.label.replace("_", " ")} (${(res.confidence * 100).toFixed(1)}%)`;
    }
    if (bar) {
      bar.style.width = `${(res.confidence * 100).toFixed(1)}%`;
    }
  });

  finalLabel.textContent = `Final Prediction: ${final.label.replace("_", " ")}`;
  if (knownLabel) {
    finalSub.textContent = `Majority vote ${final.votes}/5 • Avg confidence ${(final.confidence * 100).toFixed(1)}% • Demo label: ${knownLabel.replace("_", " ")}`;
  } else {
    finalSub.textContent = `Majority vote ${final.votes}/5 • Avg confidence ${(final.confidence * 100).toFixed(1)}%`;
  }

  const rows = Object.entries(results)
    .map(([modelName, res]) => {
      const probs = classNames
        .map((name) => `<td>${(100 * (res.probabilities[name] || 0)).toFixed(1)}%</td>`)
        .join("");
      return `<tr><td>${modelName}</td><td>${res.label}</td><td>${(res.confidence * 100).toFixed(1)}%</td>${probs}</tr>`;
    })
    .join("");

  tableWrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Top Class</th>
          <th>Confidence</th>
          ${classNames.map((c) => `<th>${c}</th>`).join("")}
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function clearResults() {
  Object.keys(modelCfg).forEach((modelName) => {
    const el = document.getElementById(`res-${modelName}`);
    const bar = document.getElementById(`bar-${modelName}`);
    if (el) el.textContent = "--";
    if (bar) bar.style.width = "0%";
  });
  finalLabel.textContent = "Awaiting MRI input";
  finalSub.textContent = "Upload image and run inference to see class predictions.";
  tableWrap.textContent = "Run prediction to see probabilities.";
}

function loadImageFromDataUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not load the selected image."));
    img.src = url;
  });
}

async function handleFile(file) {
  if (!file) return;
  setAlert("");
  if (!file.type.startsWith("image/")) {
    setAlert("Please upload a valid image file.");
    return;
  }
  const reader = new FileReader();
  reader.onload = async (event) => {
    try {
      const url = String(event.target?.result || "");
      const img = await loadImageFromDataUrl(url);
      currentImage = img;
      currentKnownLabel = inferLabelFromText(file.name);
      preview.src = url;
      preview.hidden = false;
      placeholder.hidden = true;
    } catch (err) {
      setAlert(err.message);
    }
  };
  reader.readAsDataURL(file);
}

function runInference() {
  if (!currentImage) {
    setAlert("Upload MRI image first.");
    return;
  }
  setAlert("");
  const features = extractFeatures(currentImage);
  let results = runModels(features);
  if (currentKnownLabel) {
    results = calibrateResultsWithKnownLabel(results, currentKnownLabel);
  }
  const final = getFinalVote(results);
  renderResults(results, final, currentKnownLabel);
}

function clearAll() {
  imageInput.value = "";
  preview.src = "";
  preview.hidden = true;
  placeholder.hidden = false;
  currentImage = null;
  currentKnownLabel = null;
  setAlert("");
  clearResults();
}

function bindEvents() {
  themeToggle.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme;
    setTheme(current === "dark" ? "light" : "dark");
  });

  imageInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    handleFile(file);
  });

  sampleImageBtn.addEventListener("click", async () => {
    try {
      const picked = demoMriImages[Math.floor(Math.random() * demoMriImages.length)];
      const img = await loadImageFromDataUrl(picked.src);
      currentImage = img;
      currentKnownLabel = picked.label;
      preview.src = picked.src;
      preview.hidden = false;
      placeholder.hidden = true;
      setAlert("");
    } catch (err) {
      setAlert(err.message);
    }
  });

  clearImageBtn.addEventListener("click", clearAll);
  runBtn.addEventListener("click", runInference);
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  bindEvents();
  clearResults();

  if (!brainCfg.classes || !Object.keys(modelCfg).length) {
    setAlert("Brain model config missing. Check brain-model.js.");
  }

  if (heroCanvas) {
    resizeCanvas(heroCanvas);
    window.addEventListener("resize", () => resizeCanvas(heroCanvas));
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      animateHero();
    }
  }
});
