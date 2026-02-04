/* EEG Emotion Detection UI Logic (No backend required) */

const config = window.APP_CONFIG || {};
const channelLabels = config.channelLabels || [];
const exampleValues = config.exampleValues || [];
const sampleValues = config.sampleValues || [];
const emotionLabels = config.emotionLabels || [];
const model = config.model || {};

const scalerMean = model.scalerMean || [];
const scalerScale = model.scalerScale || [];
const linearCoef = model.linearCoef || [];
const linearIntercept = model.linearIntercept || 0;
const logisticCoef = model.logisticCoef || [];
const logisticIntercept = model.logisticIntercept || [];

const inputGrid = document.getElementById("input-grid");
const form = document.getElementById("eeg-form");
const alertBox = document.getElementById("alert");
const predictBtn = document.getElementById("predict-btn");
const autofillBtn = document.getElementById("autofill-btn");
const autofillHeroBtn = document.getElementById("autofill-hero");
const clearBtn = document.getElementById("clear-btn");
const themeToggle = document.getElementById("theme-toggle");

const emotionBadge = document.getElementById("emotion-badge");
const emotionLabel = document.getElementById("emotion-label");
const emotionSubtitle = document.getElementById("emotion-subtitle");
const confidenceRing = document.getElementById("confidence-ring");
const confidenceValue = document.getElementById("confidence-value");
const intensityFill = document.getElementById("intensity-fill");
const intensityValue = document.getElementById("intensity-value");
const recentList = document.getElementById("recent-list");

const detailsBtn = document.getElementById("details-btn");
const exportJsonBtn = document.getElementById("export-json-btn");
const exportPdfBtn = document.getElementById("export-pdf-btn");
const modal = document.getElementById("modal");
const modalClose = document.getElementById("modal-close");
const modalContent = document.getElementById("modal-content");
const toastContainer = document.getElementById("toast-container");

const heroCanvas = document.getElementById("hero-canvas");
const activityCanvas = document.getElementById("activity-canvas");
const resultWaveCanvas = document.getElementById("result-wave");

let inputs = [];
let currentResult = null;
let currentValues = null;
let waveIntensity = 0.2;
let waveConfidence = 0.2;

const emotionMeta = {
  Happy: { color: "#F9D976", emoji: "😊" },
  Sad: { color: "#5B8DEF", emoji: "😢" },
  Angry: { color: "#FF6B6B", emoji: "😠" },
  Calm: { color: "#8E9BFF", emoji: "😌" },
  Fearful: { color: "#94A3B8", emoji: "😨" },
  Surprised: { color: "#F4A261", emoji: "😲" },
};

const ringCircumference = 2 * Math.PI * 50;
confidenceRing.style.strokeDasharray = String(ringCircumference);
confidenceRing.style.strokeDashoffset = String(ringCircumference);

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

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3500);
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

function setLoading(loading) {
  predictBtn.classList.toggle("loading", loading);
  predictBtn.disabled = loading;
  predictBtn.setAttribute("aria-busy", String(loading));
}

function setInvalid(input, message) {
  const wrapper = input.closest(".input-field");
  wrapper.classList.add("invalid");
  const error = wrapper.querySelector(".field-error");
  error.textContent = message;
  input.setAttribute("aria-invalid", "true");
}

function setValid(input) {
  const wrapper = input.closest(".input-field");
  wrapper.classList.remove("invalid");
  const error = wrapper.querySelector(".field-error");
  error.textContent = "";
  input.setAttribute("aria-invalid", "false");
}

function validateInput(input) {
  const value = input.value.trim();
  if (value === "") {
    setInvalid(input, "Required");
    return false;
  }
  if (!Number.isFinite(Number(value))) {
    setInvalid(input, "Enter a number");
    return false;
  }
  setValid(input);
  return true;
}

function validateAll() {
  let valid = true;
  inputs.forEach((input) => {
    if (!validateInput(input)) {
      valid = false;
    }
  });
  return valid;
}

function collectValues() {
  if (!validateAll()) {
    showToast("Fix invalid inputs before predicting.", "error");
    return null;
  }
  return inputs.map((input) => Number(input.value));
}

function getRandomSample() {
  if (sampleValues.length) {
    const pick = sampleValues[Math.floor(Math.random() * sampleValues.length)];
    if (pick && pick.length === inputs.length) {
      return pick;
    }
  }
  return inputs.map(() => Number((Math.random() * 80 - 40).toFixed(3)));
}

function autofill(values) {
  const useValues =
    values && values.length === inputs.length ? values : getRandomSample();
  inputs.forEach((input, idx) => {
    input.value = useValues[idx];
    validateInput(input);
  });
  showToast("Random EEG sample inserted.");
}

function clearAll() {
  inputs.forEach((input) => {
    input.value = "";
    setValid(input);
  });
  currentResult = null;
  currentValues = null;
  emotionBadge.textContent = "--";
  emotionLabel.textContent = "Awaiting prediction";
  emotionSubtitle.textContent = "Enter EEG values to begin.";
  confidenceValue.textContent = "0%";
  confidenceRing.style.strokeDashoffset = String(ringCircumference);
  intensityFill.style.width = "0%";
  intensityValue.textContent = "0.00 / 10";
  waveIntensity = 0.2;
  waveConfidence = 0.2;
  showToast("Inputs cleared.");
}

function updateResults(data) {
  const meta = emotionMeta[data.label] || { color: "#7f7bff", emoji: "🧠" };
  const confidencePercent = Math.round(data.confidence * 100);
  const intensityPercent = Math.min(100, (data.intensity / 10) * 100);

  emotionBadge.textContent = meta.emoji;
  emotionBadge.style.background = `linear-gradient(135deg, ${meta.color}, rgba(255,255,255,0.6))`;
  emotionLabel.textContent = data.label;
  emotionSubtitle.textContent = `Confidence ${confidencePercent}% • Intensity ${data.intensity.toFixed(
    2
  )}`;

  document.documentElement.style.setProperty("--ring", meta.color);
  confidenceValue.textContent = `${confidencePercent}%`;
  confidenceRing.style.strokeDashoffset = String(
    ringCircumference * (1 - confidencePercent / 100)
  );

  intensityFill.style.width = `${intensityPercent}%`;
  intensityValue.textContent = `${data.intensity.toFixed(2)} / 10`;

  waveIntensity = data.intensity / 10;
  waveConfidence = data.confidence;

  document
    .querySelectorAll(".card")
    .forEach((card) => card.classList.add("result-updated"));
  setTimeout(() => {
    document
      .querySelectorAll(".card")
      .forEach((card) => card.classList.remove("result-updated"));
  }, 500);
}

function saveRecent(data) {
  const entry = {
    label: data.label,
    confidence: data.confidence,
    intensity: data.intensity,
    timestamp: data.timestamp,
  };
  const existing = JSON.parse(localStorage.getItem("eeg_recent") || "[]");
  existing.unshift(entry);
  const trimmed = existing.slice(0, 6);
  localStorage.setItem("eeg_recent", JSON.stringify(trimmed));
  renderRecent(trimmed);
}

function renderRecent(items) {
  recentList.innerHTML = "";
  if (!items.length) {
    recentList.innerHTML = '<p class="muted">No predictions yet.</p>';
    return;
  }
  items.forEach((item) => {
    const meta = emotionMeta[item.label] || { color: "#7f7bff", emoji: "🧠" };
    const card = document.createElement("div");
    card.className = "recent-item";
    const date = new Date(item.timestamp);
    card.innerHTML = `
      <strong>${meta.emoji} ${item.label}</strong>
      <div class="muted">Intensity ${item.intensity.toFixed(2)} / 10</div>
      <div class="muted">Confidence ${(item.confidence * 100).toFixed(1)}%</div>
      <div class="muted">${date.toLocaleString()}</div>
    `;
    recentList.appendChild(card);
  });
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const total = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / total);
}

function predictLocal(values) {
  if (
    values.length !== scalerMean.length ||
    scalerMean.length !== scalerScale.length
  ) {
    throw new Error("Model not initialized correctly.");
  }

  const scaled = values.map(
    (v, i) => (v - scalerMean[i]) / (scalerScale[i] || 1)
  );
  const intensityRaw = dot(scaled, linearCoef) + linearIntercept;
  const intensity = Math.min(10, Math.max(0, intensityRaw));

  const logits = logisticCoef.map(
    (coefRow, idx) => dot(scaled, coefRow) + (logisticIntercept[idx] || 0)
  );
  const probs = softmax(logits);
  const maxIndex = probs.indexOf(Math.max(...probs));

  const label = emotionLabels[maxIndex] || "Unknown";
  const confidence = probs[maxIndex] || 0;
  const probMap = {};
  probs.forEach((value, idx) => {
    probMap[emotionLabels[idx] || `Class ${idx}`] = value;
  });

  return {
    label,
    confidence,
    intensity,
    probabilities: probMap,
    timestamp: new Date().toISOString(),
  };
}

function runPrediction(values) {
  setLoading(true);
  try {
    const data = predictLocal(values);
    currentResult = data;
    currentValues = values;
    updateResults(data);
    saveRecent(data);
    showToast("Prediction ready.");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    setLoading(false);
  }
}

function openModal() {
  if (!currentResult) {
    showToast("Run a prediction first.", "error");
    return;
  }
  const probs = currentResult.probabilities || {};
  const rows = Object.entries(probs)
    .map(
      ([label, value]) =>
        `<div><strong>${label}</strong>: ${(value * 100).toFixed(1)}%</div>`
    )
    .join("");
  modalContent.innerHTML = `
    <div><strong>Emotion:</strong> ${currentResult.label}</div>
    <div><strong>Intensity:</strong> ${currentResult.intensity.toFixed(2)} / 10</div>
    <div><strong>Confidence:</strong> ${(currentResult.confidence * 100).toFixed(
      1
    )}%</div>
    <div><strong>Class probabilities</strong></div>
    ${rows}
  `;
  modal.classList.add("active");
  modal.setAttribute("aria-hidden", "false");
}

function closeModal() {
  modal.classList.remove("active");
  modal.setAttribute("aria-hidden", "true");
}

function exportJson() {
  if (!currentResult) {
    showToast("No prediction to export.", "error");
    return;
  }
  const payload = {
    result: currentResult,
    values: currentValues,
    channels: channelLabels,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "eeg-prediction.json";
  link.click();
  URL.revokeObjectURL(url);
  showToast("JSON exported.");
}

function exportPdf() {
  if (!currentResult) {
    showToast("No prediction to export.", "error");
    return;
  }
  window.print();
}

function attachInputListeners() {
  inputs.forEach((input) => {
    input.addEventListener("input", () => validateInput(input));
    input.addEventListener("blur", () => validateInput(input));
  });
}

function bindEvents() {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const values = collectValues();
    if (!values) return;
    runPrediction(values);
  });

  autofillBtn.addEventListener("click", () => {
    const values = getRandomSample();
    autofill(values);
    runPrediction(values);
  });
  if (autofillHeroBtn) {
    autofillHeroBtn.addEventListener("click", () => {
      const values = getRandomSample();
      autofill(values);
      runPrediction(values);
    });
  }
  clearBtn.addEventListener("click", clearAll);
  detailsBtn.addEventListener("click", openModal);
  modalClose.addEventListener("click", closeModal);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });
  exportJsonBtn.addEventListener("click", exportJson);
  exportPdfBtn.addEventListener("click", exportPdf);

  themeToggle.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme;
    setTheme(current === "dark" ? "light" : "dark");
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      clearAll();
    }
  });
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * ratio;
  canvas.height = canvas.clientHeight * ratio;
}

function drawWave(ctx, time, intensity, colorStops) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  colorStops.forEach(([stop, color]) => gradient.addColorStop(stop, color));
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);

  ctx.beginPath();
  for (let x = 0; x <= width; x += 8) {
    const y =
      height / 2 +
      Math.sin(x * 0.012 + time) * (height * 0.2 * intensity) +
      Math.sin(x * 0.02 + time * 1.4) * (height * 0.08);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function animateHero() {
  if (!heroCanvas) return;
  const ctx = heroCanvas.getContext("2d");
  const animate = (time) => {
    const t = time * 0.001;
    drawWave(ctx, t, 0.6, [
      [0, "rgba(127, 123, 255, 0.6)"],
      [0.5, "rgba(76, 201, 240, 0.4)"],
      [1, "rgba(94, 234, 212, 0.4)"],
    ]);
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);
}

function animateResultWaves() {
  if (!activityCanvas || !resultWaveCanvas) return;
  const ctxActivity = activityCanvas.getContext("2d");
  const ctxResult = resultWaveCanvas.getContext("2d");
  const animate = (time) => {
    const t = time * 0.001;
    drawWave(ctxActivity, t, 0.5 + waveConfidence * 0.5, [
      [0, "rgba(127, 123, 255, 0.6)"],
      [1, "rgba(76, 201, 240, 0.5)"],
    ]);
    drawWave(ctxResult, t, 0.4 + waveIntensity * 0.6, [
      [0, "rgba(94, 234, 212, 0.7)"],
      [1, "rgba(127, 123, 255, 0.5)"],
    ]);
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);
}

function initCanvases() {
  if (heroCanvas) resizeCanvas(heroCanvas);
  if (activityCanvas) resizeCanvas(activityCanvas);
  if (resultWaveCanvas) resizeCanvas(resultWaveCanvas);
  window.addEventListener("resize", () => {
    if (heroCanvas) resizeCanvas(heroCanvas);
    if (activityCanvas) resizeCanvas(activityCanvas);
    if (resultWaveCanvas) resizeCanvas(resultWaveCanvas);
  });
}

function initRecent() {
  const existing = JSON.parse(localStorage.getItem("eeg_recent") || "[]");
  renderRecent(existing);
}

function buildInputGrid() {
  inputGrid.innerHTML = "";
  channelLabels.forEach((label, idx) => {
    const wrapper = document.createElement("div");
    wrapper.className = "input-field";
    wrapper.innerHTML = `
      <label for="ch${idx + 1}">
        <span class="channel-name">${label}</span>
        <span class="channel-index">Ch ${idx + 1}</span>
      </label>
      <input
        id="ch${idx + 1}"
        data-channel-index="${idx}"
        name="ch${idx + 1}"
        type="number"
        step="any"
        inputmode="decimal"
        autocomplete="off"
        placeholder="0.000"
        aria-label="EEG channel ${label}"
        required
      />
      <span class="field-error" aria-live="polite"></span>
    `;
    inputGrid.appendChild(wrapper);
  });
  inputs = Array.from(document.querySelectorAll(".input-field input"));
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  buildInputGrid();
  attachInputListeners();
  bindEvents();
  initCanvases();
  initRecent();

  if (!model || !linearCoef.length) {
    setAlert("Model data missing. Please regenerate model.js.");
  }

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduceMotion) {
    animateHero();
    animateResultWaves();
  }
});
