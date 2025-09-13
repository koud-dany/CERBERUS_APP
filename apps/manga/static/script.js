// ------------------ Workflow Selection ------------------
function selectWorkflow(type) {
    const extractCard = document.getElementById("uploadCard");
    const directCard = document.getElementById("directInputCard");

    document.getElementById("extractWorkflow").classList.remove("selected");
    document.getElementById("directWorkflow").classList.remove("selected");

    if (type === "extract") {
        document.getElementById("extractWorkflow").classList.add("selected");
        extractCard.style.display = "block";
        directCard.style.display = "none";
    } else {
        document.getElementById("directWorkflow").classList.add("selected");
        directCard.style.display = "block";
        extractCard.style.display = "none";
    }
}

// ------------------ Upload Handling ------------------
let uploadedFiles = [];
let currentSessionId = null;

async function handleUpload(inputId, previewId, progressId, alertId) {
    const files = document.getElementById(inputId).files;
    if (!files.length) return;

    const formData = new FormData();
    for (let file of files) {
        formData.append("files", file);
    }

    const progressBar = document.getElementById(progressId);
    progressBar.classList.remove("hidden");

    try {
        const res = await fetch("/manga/upload", {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            uploadedFiles = data.files;
            currentSessionId = data.session_id;

            // Preview thumbnails
            const preview = document.getElementById(previewId);
            preview.innerHTML = "";
            uploadedFiles.forEach(file => {
                const div = document.createElement("div");
                div.classList.add("image-item");
                div.innerHTML = `<img src="${file.url}"><div class="image-info"><h4>${file.filename}</h4></div>`;
                preview.appendChild(div);
            });

            document.getElementById(alertId).innerHTML = `<div class="alert alert-success">${data.message}</div>`;
            document.getElementById("extractCard").style.display = "block";
        } else {
            document.getElementById(alertId).innerHTML = `<div class="alert alert-error">${data.error}</div>`;
        }
    } catch (err) {
        document.getElementById(alertId).innerHTML = `<div class="alert alert-error">Upload failed: ${err}</div>`;
    } finally {
        progressBar.classList.add("hidden");
    }
}

// ------------------ Extract Text ------------------
async function extractText() {
    if (!uploadedFiles.length) {
        alert("Please upload files first!");
        return;
    }

    document.getElementById("extractLoading").classList.add("show");

    try {
        const res = await fetch("/manga/extract-text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ files: uploadedFiles })
        });
        const data = await res.json();

        const resultsDiv = document.getElementById("textResults");
        resultsDiv.innerHTML = "";
        if (data.results) {
            data.results.forEach(r => {
                const item = document.createElement("div");
                item.classList.add("image-item");
                item.innerHTML = `
                    <img src="${r.url}">
                    <div class="image-info">
                        <h4>${r.filename}</h4>
                        <textarea class="form-control editable-text">${r.text}</textarea>
                    </div>`;
                resultsDiv.appendChild(item);
            });
        }
        document.getElementById("recapCard").style.display = "block";
    } catch (err) {
        alert("Error extracting text: " + err);
    } finally {
        document.getElementById("extractLoading").classList.remove("show");
    }
}

// ------------------ Generate Recap ------------------
async function generateRecap() {
    const textAreas = document.querySelectorAll("#textResults textarea");
    const texts = Array.from(textAreas).map(t => t.value);
    const prompt = document.getElementById("customPrompt").value;

    document.getElementById("recapLoading").classList.add("show");

    try {
        const res = await fetch("/manga/generate-recap", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ texts, prompt })
        });
        const data = await res.json();

        if (data.recap) {
            document.getElementById("recapResult").innerHTML = `
                <div class="recap-output">${data.recap.replace(/\n/g, '<br>')}</div>
            `;
            document.getElementById("videoCard").style.display = "block";
        } else {
            document.getElementById("recapResult").innerHTML = `<div class="alert alert-error">${data.error}</div>`;
        }
    } catch (err) {
        document.getElementById("recapResult").innerHTML = `<div class="alert alert-error">Error: ${err}</div>`;
    } finally {
        document.getElementById("recapLoading").classList.remove("show");
    }
}

// ------------------ File Input Helpers ------------------
function triggerFileInput(inputId) {
    document.getElementById(inputId).click();
}

// ------------------ Direct Recap Processing ------------------
async function processDirectRecap() {
    if (!uploadedFiles.length) {
        alert("Please upload files first!");
        return;
    }

    const prompt = document.getElementById("directCustomPrompt").value;
    document.getElementById("directRecapLoading").classList.add("show");

    try {
        // First extract text
        const extractRes = await fetch("/manga/extract-text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ files: uploadedFiles })
        });
        const extractData = await extractRes.json();
        
        if (!extractRes.ok) {
            throw new Error(extractData.error || "Failed to extract text");
        }

        const texts = extractData.results.map(r => r.text);

        // Then generate recap
        const recapRes = await fetch("/manga/generate-recap", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ texts, prompt })
        });
        const recapData = await recapRes.json();

        if (recapRes.ok) {
            document.getElementById("directRecapSection").style.display = "block";
            document.getElementById("directRecapResult").innerHTML = `
                <div class="recap-output">${recapData.recap.replace(/\n/g, '<br>')}</div>
                <button class="btn btn-success" onclick="proceedToVideo()">🎬 Create Video</button>
            `;
            document.getElementById("videoCard").style.display = "block";
        } else {
            throw new Error(recapData.error || "Failed to generate recap");
        }
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        document.getElementById("directRecapLoading").classList.remove("show");
    }
}

function proceedToVideo() {
    document.getElementById("videoCard").scrollIntoView({ behavior: 'smooth' });
}

// ------------------ Video Creation ------------------
async function createOpenAIVideo() {
    alert("OpenAI Video creation functionality - implement based on your needs");
}

async function createLiveVideo() {
    alert("Live Video creation functionality - implement based on your needs");
}

async function createAutomatedVideo() {
    alert("Automated Video creation functionality - implement based on your needs");
}
