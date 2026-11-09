// Classifier page: choose/drop an image, send it to /api/classify, show both models' answers side by side.

const MODELS = ["baseline", "optimized"];

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const dropzoneText = document.getElementById("dropzone-text");
const classifyButton = document.getElementById("classify-button");
const errorBox = document.getElementById("error");
const verdict = document.getElementById("verdict");

let selectedFile = null;

const percent = (value) => `${(value * 100).toFixed(1)}%`;
const panel = (name) => document.getElementById(name);
const part = (name, key) => panel(name).querySelector(`[data-${key}]`);

function showError(box, message) {
    box.textContent = message;
    box.hidden = !message;
}

function clearResults() {
    verdict.hidden = true;
    for (const name of MODELS) {
        part(name, "prediction").hidden = true;
        showError(part(name, "error"), "");
    }
}

function selectFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
        showError(errorBox, "Please choose an image file.");
        return;
    }
    selectedFile = file;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    dropzoneText.hidden = true;
    classifyButton.disabled = false;
    showError(errorBox, "");
    clearResults();
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
    selectFile(event.dataTransfer.files[0]);
});

function renderBars(list, items) {
    list.replaceChildren();
    items.slice(0, 5).forEach((item, index) => {
        const row = document.createElement("li");
        if (index === 0) row.className = "best";

        const label = document.createElement("span");
        label.textContent = item.label;
        const track = document.createElement("span");
        track.className = "track";
        const fill = document.createElement("span");
        fill.className = "fill";
        fill.style.width = percent(item.probability);
        track.append(fill);
        const value = document.createElement("span");
        value.className = "value";
        value.textContent = percent(item.probability);

        row.append(label, track, value);
        list.append(row);
    });
}

function renderModel(name, output) {
    if (output.error) {
        showError(part(name, "error"), output.error);
        return;
    }
    const info = output.model;
    part(name, "meta").textContent =
        `Test accuracy ${percent(info.test_accuracy)} · ${info.parameters.toLocaleString("en-US")} parameters · ${output.predict_ms} ms`;
    part(name, "label").textContent = output.label;
    part(name, "confidence").textContent = `${percent(output.confidence)} confident`;
    part(name, "model-input").src = output.model_input;
    renderBars(part(name, "bars"), output.probabilities);
    part(name, "prediction").hidden = false;
}

function describeLayers(config) {
    if (config.hidden_sizes) return config.hidden_sizes.join(" → ");
    return Object.entries(config.architecture ?? {}).map(([k, v]) => `${k} ${v}`).join(", ");
}

// Hyperparameters of both models in one table, differences highlighted
function renderDetails(results) {
    const configs = MODELS.map((name) => results[name]?.model?.config);
    if (configs.some((config) => !config)) return;

    const rows = [
        ["Hidden layers", (c) => describeLayers(c)],
        ["Dropout", (c) => c.dropout.toFixed(2)],
        ["Optimizer", (c) => c.optimizer + (c.optimizer === "SGD" && c.momentum ? ` (momentum ${c.momentum.toFixed(2)})` : "")],
        ["Learning rate", (c) => c.learning_rate.toPrecision(3)],
        ["Weight decay", (c) => (c.weight_decay ? c.weight_decay.toExponential(1) : "0")],
        ["Batch size", (c) => c.batch_size],
        ["Epochs", (c) => c.epochs],
        ["Test accuracy", (_, i) => percent(results[MODELS[i]].model.test_accuracy)],
    ];
    // CIFAR10 only: data augmentation is one of the tuned hyperparameters
    if (configs.some((config) => "augment" in config)) {
        rows.splice(rows.length - 1, 0, ["Augmentation", (c) => (c.augment ? "flip + crop" : "none")]);
    }
    const body = document.getElementById("details-body");
    body.replaceChildren();
    for (const [title, format] of rows) {
        const row = document.createElement("tr");
        const values = configs.map((config, i) => String(format(config, i)));
        if (values[0] !== values[1]) row.className = "changed";
        for (const text of [title, ...values]) {
            const cell = document.createElement("td");
            cell.textContent = text;
            row.append(cell);
        }
        body.append(row);
    }
    document.getElementById("details").hidden = false;
}

function renderVerdict(results) {
    const [a, b] = MODELS.map((name) => results[name]);
    if (a.error || b.error) return;
    verdict.textContent = a.label === b.label
        ? `Both models say ${a.label}.`
        : `The models disagree: baseline says ${a.label}, optimized says ${b.label}.`;
    verdict.className = `verdict ${a.label === b.label ? "agree" : "disagree"}`;
    verdict.hidden = false;
}

classifyButton.addEventListener("click", async () => {
    if (!selectedFile) return;

    classifyButton.disabled = true;
    classifyButton.textContent = "Classifying…";
    showError(errorBox, "");
    clearResults();

    const form = new FormData();
    form.append("image", selectedFile);

    try {
        const response = await fetch("/api/classify", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Classification failed.");
        MODELS.forEach((name) => renderModel(name, data[name]));
        renderVerdict(data);
        renderDetails(data);
    } catch (error) {
        showError(errorBox, error.message);
    } finally {
        classifyButton.disabled = false;
        classifyButton.textContent = "Classify with both models";
    }
});

// Model status: Cog reports STARTING while setup() loads the weights, then READY
async function checkModels() {
    let statuses = {};
    try {
        statuses = await (await fetch("/api/health")).json();
    } catch {
        // the website itself is down; every model shows as not reachable
    }
    const states = {
        READY: ["ready", "Ready"],
        STARTING: ["starting", "Loading…"],
        BUSY: ["ready", "Busy"],
    };
    for (const name of MODELS) {
        const [className, text] = states[statuses[name]] ?? ["down", "Not reachable"];
        part(name, "status").className = `status ${className}`;
        part(name, "status-text").textContent = text;
    }
    const allReady = MODELS.every((name) => statuses[name] === "READY");
    setTimeout(checkModels, allReady ? 15000 : 3000);
}
checkModels();
