// app.js - Lightweight Frontend Controller for Tamil Handwriting OCR

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Input
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const replaceImgBtn = document.getElementById('replace-img-btn');
    const removeImgBtn = document.getElementById('remove-img-btn');
    const recognizeDocBtn = document.getElementById('recognize-doc-btn');
    const uploadSpinner = document.getElementById('upload-spinner');
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');
    const preflightAlert = document.getElementById('preflight-alert');
    const preflightTitle = document.getElementById('preflight-title');
    const preflightMessage = document.getElementById('preflight-message');

    // DOM Elements - Output
    const emptyState = document.getElementById('empty-state');
    const resultDisplay = document.getElementById('result-display');
    const statsContainer = document.getElementById('stats-container');
    const statLines = document.getElementById('stat-lines');
    const statWords = document.getElementById('stat-words');
    const tamilEditor = document.getElementById('tamil-editor');

    // Action Buttons
    const copyBtn = document.getElementById('copy-btn');
    const copyLabel = document.getElementById('copy-label');
    const exportTxtBtn = document.getElementById('export-txt-btn');
    const exportDocxBtn = document.getElementById('export-docx-btn');
    const exportPdfBtn = document.getElementById('export-pdf-btn');

    let selectedFile = null;

    // Browse & Replace triggers
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    if (replaceImgBtn) {
        replaceImgBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    dropzone.addEventListener('click', () => {
        if (!selectedFile) fileInput.click();
    });

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetInput();
    });

    function handleFileSelection(file) {
        hideError();
        hidePreflight();
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            showError('Please select a valid image file (PNG, JPG, JPEG, BMP).');
            return;
        }

        const maxBytes = 10 * 1024 * 1024;
        if (file.size > maxBytes) {
            showError('Selected image exceeds the 10 MB maximum limit.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            recognizeDocBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        // Run non-blocking pre-flight image quality check
        runPreflightCheck(file);
    }

    async function runPreflightCheck(file) {
        try {
            const formData = new FormData();
            formData.append('image', file);
            const response = await fetch('/api/ocr/preflight', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) return;
            const data = await response.json();
            if (!data.success) return;

            if (data.quality_status === 'severe_low_res') {
                showPreflight(
                    'Very small handwriting detected',
                    data.warning || 'Some text lines are very small, so Tamil characters or vowel marks may be misrecognized. For better results, take a closer, well-lit photograph. You can still continue with this image.'
                );
            } else if (data.quality_status === 'caution_low_res') {
                showPreflight(
                    'Low-resolution handwriting detected',
                    data.warning || 'Some text lines are small, so Tamil characters or vowel marks may be misrecognized. For better results, take a closer, well-lit photograph. You can still continue with this image.'
                );
            } else if (data.quality_status === 'no_text_detected') {
                showPreflight(
                    'No clear text lines detected',
                    data.warning || 'No clear text lines detected. Please ensure the document is clear and well-lit. You can still continue with this image.'
                );
            } else {
                hidePreflight();
            }
        } catch (e) {
            // Advisory pre-flight check should never block normal OCR execution
            console.debug('Pre-flight check skipped:', e);
        }
    }

    function showPreflight(title, msg) {
        if (!preflightAlert) return;
        preflightTitle.textContent = title;
        preflightMessage.textContent = msg;
        preflightAlert.classList.remove('hidden');
    }

    function hidePreflight() {
        if (!preflightAlert) return;
        preflightAlert.classList.add('hidden');
    }

    function resetInput() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');
        recognizeDocBtn.disabled = true;
        hideError();
        hidePreflight();
    }

    // Recognize Text Execution
    recognizeDocBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        setLoading(true);
        hideError();

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/api/ocr/document', {
                method: 'POST',
                body: formData
            });

            const resText = await response.text();
            let data;
            try {
                data = JSON.parse(resText);
            } catch (jsonErr) {
                const cleanMsg = resText.replace(/<[^>]*>?/gm, ' ').replace(/\s+/g, ' ').trim();
                throw new Error(cleanMsg || `Server error (HTTP ${response.status})`);
            }

            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Failed to process document.');
            }

            displayResults(data);
        } catch (err) {
            showError(err.message || 'An error occurred during recognition.');
        } finally {
            setLoading(false);
        }
    });

    function displayResults(data) {
        emptyState.classList.add('hidden');
        resultDisplay.classList.remove('hidden');
        statsContainer.classList.remove('hidden');

        statLines.textContent = data.total_lines || 0;
        statWords.textContent = data.total_words || 0;
        tamilEditor.value = data.full_text || '';
    }

    function setLoading(isLoading) {
        recognizeDocBtn.disabled = isLoading;
        if (isLoading) {
            uploadSpinner.classList.remove('hidden');
            recognizeDocBtn.querySelector('.btn-text').textContent = 'Recognizing Text...';
        } else {
            uploadSpinner.classList.add('hidden');
            recognizeDocBtn.querySelector('.btn-text').textContent = 'Recognize Text';
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.classList.remove('hidden');
    }

    function hideError() {
        errorAlert.classList.add('hidden');
    }

    // Copy to Clipboard with Visual Feedback
    copyBtn.addEventListener('click', async () => {
        const text = tamilEditor.value;
        if (!text) return;

        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                tamilEditor.select();
                document.execCommand('copy');
            }
            copyLabel.textContent = 'Copied!';
            setTimeout(() => {
                copyLabel.textContent = 'Copy Text';
            }, 2000);
        } catch (err) {
            showError('Unable to copy text to clipboard.');
        }
    });

    // File Exports (TXT, DOCX, PDF)
    async function triggerExport(url, defaultFilename) {
        const text = tamilEditor.value;
        if (!text) {
            showError('No recognized text available for export.');
            return;
        }

        try {
            const resp = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!resp.ok) {
                const errJson = await resp.json().catch(() => ({}));
                throw new Error(errJson.error || `Export failed (HTTP ${resp.status})`);
            }

            const blob = await resp.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = downloadUrl;
            a.download = defaultFilename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(downloadUrl);
            a.remove();
        } catch (e) {
            showError(e.message || 'File export failed.');
        }
    }

    exportTxtBtn.addEventListener('click', () => {
        triggerExport('/api/export/txt', 'tamil_handwritten_ocr.txt');
    });

    exportDocxBtn.addEventListener('click', () => {
        triggerExport('/api/export/docx', 'tamil_handwritten_ocr.docx');
    });

    exportPdfBtn.addEventListener('click', () => {
        triggerExport('/api/export/pdf', 'tamil_handwritten_ocr.pdf');
    });
});
