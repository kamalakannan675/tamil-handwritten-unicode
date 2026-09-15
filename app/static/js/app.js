// app.js - Frontend logic for Tamil Handwritten Character Recognition

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching elements
    const tabBtnUpload = document.getElementById('tab-btn-upload');
    const tabBtnDraw = document.getElementById('tab-btn-draw');
    const panelUpload = document.getElementById('panel-upload');
    const panelDraw = document.getElementById('panel-draw');

    // Upload elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeImgBtn = document.getElementById('remove-img-btn');
    const recognizeUploadBtn = document.getElementById('recognize-upload-btn');
    const uploadSpinner = document.getElementById('upload-spinner');

    // Canvas elements
    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');
    const clearCanvasBtn = document.getElementById('clear-canvas-btn');
    const recognizeDrawBtn = document.getElementById('recognize-draw-btn');
    const drawSpinner = document.getElementById('draw-spinner');

    // Result elements
    const emptyState = document.getElementById('empty-state');
    const resultDisplay = document.getElementById('result-display');
    const resGlyph = document.getElementById('res-glyph');
    const resCodepoints = document.getElementById('res-codepoints');
    const resClassId = document.getElementById('res-class-id');
    const resConfidence = document.getElementById('res-confidence');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceLevelBadge = document.getElementById('confidence-level-badge');
    const lowConfAlert = document.getElementById('low-conf-alert');
    const top3List = document.getElementById('top3-list');
    const tamilEditor = document.getElementById('tamil-editor');

    // Export & Action elements
    const copyBtn = document.getElementById('copy-btn');
    const copyLabel = document.getElementById('copy-label');
    const exportTxtBtn = document.getElementById('export-txt-btn');
    const exportDocxBtn = document.getElementById('export-docx-btn');
    const exportPdfBtn = document.getElementById('export-pdf-btn');

    // History elements
    const historyList = document.getElementById('history-list');
    const historyEmpty = document.getElementById('history-empty');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    // Error alert
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');

    let selectedFile = null;
    let isDrawing = false;
    let hasDrawn = false;
    let sessionHistory = [];

    // ----------------------------------------------------
    // 1. Tab Navigation
    // ----------------------------------------------------
    tabBtnUpload.addEventListener('click', () => {
        tabBtnUpload.classList.add('active');
        tabBtnUpload.setAttribute('aria-selected', 'true');
        tabBtnDraw.classList.remove('active');
        tabBtnDraw.setAttribute('aria-selected', 'false');
        panelUpload.classList.add('active');
        panelDraw.classList.remove('active');
        hideError();
    });

    tabBtnDraw.addEventListener('click', () => {
        tabBtnDraw.classList.add('active');
        tabBtnDraw.setAttribute('aria-selected', 'true');
        tabBtnUpload.classList.remove('active');
        tabBtnUpload.setAttribute('aria-selected', 'false');
        panelDraw.classList.add('active');
        panelUpload.classList.remove('active');
        hideError();
    });

    // ----------------------------------------------------
    // 2. Drag & Drop Image Upload
    // ----------------------------------------------------
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener('click', () => {
        if (!selectedFile) fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearUploadedFile();
    });

    function handleSelectedFile(file) {
        const validExtensions = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
        const nameExt = file.name.split('.').pop().toLowerCase();

        if (!validExtensions.includes(file.type) && !['png', 'jpg', 'jpeg', 'bmp'].includes(nameExt)) {
            showError('Invalid file type. Please upload a PNG, JPG, JPEG, or BMP image.');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showError('File is too large. Maximum size is 10 MB.');
            return;
        }

        hideError();
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            recognizeUploadBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearUploadedFile() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');
        recognizeUploadBtn.disabled = true;
    }

    // ----------------------------------------------------
    // 3. HTML5 Canvas Drawing
    // ----------------------------------------------------
    function initCanvas() {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 14;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#000000';
    }
    initCanvas();

    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        hasDrawn = true;
        const coords = getCanvasCoordinates(e);
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const coords = getCanvasCoordinates(e);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
    }

    function stopDrawing() {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.closePath();
    }

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    // Touch events for mobile/tablets
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startDrawing(e);
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        draw(e);
    }, { passive: false });

    canvas.addEventListener('touchend', stopDrawing);

    clearCanvasBtn.addEventListener('click', () => {
        initCanvas();
        hasDrawn = false;
        hideError();
    });

    // ----------------------------------------------------
    // 4. API Prediction Calls
    // ----------------------------------------------------
    recognizeUploadBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        setLoading(true, 'upload');
        hideError();

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            let data;
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                await response.text();
                if (!response.ok) {
                    showError(`Recognition failed: Server returned HTTP ${response.status} (${response.statusText || 'Error'}).`);
                } else {
                    showError('Recognition failed: Server returned non-JSON response.');
                }
                return;
            }

            if (!response.ok || !data.success) {
                showError(data.error || `Recognition failed with HTTP ${response.status}.`);
                return;
            }

            displayResult(data);
        } catch (err) {
            showError(`Network connection error: ${err.message}`);
        } finally {
            setLoading(false, 'upload');
        }
    });

    recognizeDrawBtn.addEventListener('click', async () => {
        if (!hasDrawn) {
            showError('Please draw a Tamil character on the canvas before recognizing.');
            return;
        }

        setLoading(true, 'draw');
        hideError();

        const base64Image = canvas.toDataURL('image/png');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
            });

            let data;
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                await response.text();
                if (!response.ok) {
                    showError(`Drawing recognition failed: Server returned HTTP ${response.status} (${response.statusText || 'Error'}).`);
                } else {
                    showError('Drawing recognition failed: Server returned non-JSON response.');
                }
                return;
            }

            if (!response.ok || !data.success) {
                showError(data.error || `Drawing recognition failed with HTTP ${response.status}.`);
                return;
            }

            displayResult(data);
        } catch (err) {
            showError(`Network connection error: ${err.message}`);
        } finally {
            setLoading(false, 'draw');
        }
    });

    function getConfidenceTier(pct) {
        if (pct >= 80.0) {
            return { label: 'High Confidence', className: 'badge-confidence-high' };
        } else if (pct >= 50.0) {
            return { label: 'Medium Confidence', className: 'badge-confidence-medium' };
        } else {
            return { label: 'Low Confidence', className: 'badge-confidence-low' };
        }
    }

    function updateConfidenceBadge(pct) {
        const tier = getConfidenceTier(pct);
        if (confidenceLevelBadge) {
            confidenceLevelBadge.textContent = tier.label;
            confidenceLevelBadge.className = `badge ${tier.className}`;
        }
        if (lowConfAlert) {
            if (pct < 50.0) {
                lowConfAlert.classList.remove('hidden');
            } else {
                lowConfAlert.classList.add('hidden');
            }
        }
    }

    // ----------------------------------------------------
    // 5. Render Recognition Results
    // ----------------------------------------------------
    function displayResult(data) {
        const pred = data.prediction;
        const top3 = data.top3 || [];

        emptyState.classList.add('hidden');
        resultDisplay.classList.remove('hidden');

        // Main Result
        resGlyph.textContent = pred.unicode;
        resCodepoints.textContent = pred.codepoints;
        resClassId.textContent = pred.class_id;
        resConfidence.textContent = `${pred.confidence_percent}%`;
        confidenceBar.style.width = `${Math.min(100, pred.confidence_percent)}%`;
        updateConfidenceBadge(pred.confidence_percent);

        // Prepopulate Editable Textarea
        tamilEditor.value = pred.unicode;

        // Render Top 3 Alternative Cards
        top3List.innerHTML = '';
        top3.forEach((item, idx) => {
            const card = document.createElement('button');
            card.type = 'button';
            card.className = `top3-item ${idx === 0 ? 'active-selection' : ''}`;
            card.title = `Click to select alternative: ${item.unicode}`;
            card.innerHTML = `
                <div class="top3-rank-badge">#${item.rank} Alternative</div>
                <div class="top3-char">${item.unicode}</div>
                <div class="top3-conf-pill">${item.confidence_percent}%</div>
                <div class="top3-select-btn">${idx === 0 ? '✓ Current' : 'Choose'}</div>
            `;
            // Clicking alternative glyph replaces editor text and updates view
            card.addEventListener('click', () => {
                document.querySelectorAll('.top3-item').forEach(c => {
                    c.classList.remove('active-selection');
                    const b = c.querySelector('.top3-select-btn');
                    if (b) b.textContent = 'Choose';
                });
                card.classList.add('active-selection');
                const b = card.querySelector('.top3-select-btn');
                if (b) b.textContent = '✓ Current';

                tamilEditor.value = item.unicode;
                resGlyph.textContent = item.unicode;
                resCodepoints.textContent = item.codepoints;
                resClassId.textContent = item.class_id;
                resConfidence.textContent = `${item.confidence_percent}%`;
                confidenceBar.style.width = `${Math.min(100, item.confidence_percent)}%`;
                updateConfidenceBadge(item.confidence_percent);
            });
            top3List.appendChild(card);
        });

        // Add to Session History
        addToHistory(pred);
    }

    function addToHistory(pred) {
        sessionHistory.unshift(pred);
        if (sessionHistory.length > 10) sessionHistory.pop();

        historyEmpty.classList.add('hidden');
        renderHistory();
    }

    function renderHistory() {
        historyList.innerHTML = '';
        sessionHistory.forEach((item) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'history-chip';
            chip.innerHTML = `
                <span class="history-glyph">${item.unicode}</span>
                <span>(${item.confidence_percent}%)</span>
            `;
            chip.addEventListener('click', () => {
                tamilEditor.value = item.unicode;
            });
            historyList.appendChild(chip);
        });
    }

    clearHistoryBtn.addEventListener('click', () => {
        sessionHistory = [];
        historyList.innerHTML = '';
        historyList.appendChild(historyEmpty);
        historyEmpty.classList.remove('hidden');
    });

    // ----------------------------------------------------
    // 6. Copy & Export Actions
    // ----------------------------------------------------
    copyBtn.addEventListener('click', async () => {
        const text = tamilEditor.value;
        if (!text) return;

        try {
            await navigator.clipboard.writeText(text);
            copyLabel.textContent = 'Copied!';
            setTimeout(() => {
                copyLabel.textContent = 'Copy Unicode';
            }, 2000);
        } catch (err) {
            // Fallback
            tamilEditor.select();
            document.execCommand('copy');
            copyLabel.textContent = 'Copied!';
            setTimeout(() => {
                copyLabel.textContent = 'Copy Unicode';
            }, 2000);
        }
    });

    async function triggerExport(endpoint, fallbackFilename) {
        const text = tamilEditor.value;
        if (!text) {
            showError('No text to export. Please recognize or type a character first.');
            return;
        }

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Export failed');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fallbackFilename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (err) {
            showError(`Export error: ${err.message}`);
        }
    }

    exportTxtBtn.addEventListener('click', () => {
        triggerExport('/api/export/txt', 'tamil_text.txt');
    });

    exportDocxBtn.addEventListener('click', () => {
        triggerExport('/api/export/docx', 'tamil_document.docx');
    });

    exportPdfBtn.addEventListener('click', () => {
        triggerExport('/api/export/pdf', 'tamil_document.pdf');
    });

    // ----------------------------------------------------
    // Helper Utilities
    // ----------------------------------------------------
    function setLoading(isLoading, source) {
        if (source === 'upload') {
            recognizeUploadBtn.disabled = isLoading;
            uploadSpinner.classList.toggle('hidden', !isLoading);
        } else if (source === 'draw') {
            recognizeDrawBtn.disabled = isLoading;
            drawSpinner.classList.toggle('hidden', !isLoading);
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.classList.remove('hidden');
    }

    function hideError() {
        errorAlert.classList.add('hidden');
    }
});
