// app.js - Full-Page Tamil Handwritten Document OCR Frontend

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeImgBtn = document.getElementById('remove-img-btn');
    const recognizeDocBtn = document.getElementById('recognize-doc-btn');
    const uploadSpinner = document.getElementById('upload-spinner');

    // Results & editor
    const emptyState = document.getElementById('empty-state');
    const resultDisplay = document.getElementById('result-display');
    const tamilEditor = document.getElementById('tamil-editor');
    const statLines = document.getElementById('stat-lines');
    const statWords = document.getElementById('stat-words');
    const linesBreakdownList = document.getElementById('lines-breakdown-list');

    // Exports
    const copyBtn = document.getElementById('copy-btn');
    const copyLabel = document.getElementById('copy-label');
    const exportTxtBtn = document.getElementById('export-txt-btn');
    const exportDocxBtn = document.getElementById('export-docx-btn');
    const exportPdfBtn = document.getElementById('export-pdf-btn');

    // Error alert
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');

    let selectedFile = null;

    // Browse button handler
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener('click', () => {
        if (!selectedFile) fileInput.click();
    });

    // Drag & drop handlers
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
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFileInput();
    });

    function handleFileSelection(file) {
        hideError();
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            showError('Please select a valid image file (PNG, JPG, JPEG, BMP).');
            return;
        }

        const maxBytes = 10 * 1024 * 1024;
        if (file.size > maxBytes) {
            showError('Selected image exceeds the 10 MB maximum file size.');
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
    }

    function resetFileInput() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');
        recognizeDocBtn.disabled = true;
        hideError();
    }

    // Recognize Document Handler
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
            showError(err.message || 'An error occurred during OCR inference.');
        } finally {
            setLoading(false);
        }
    });

    function displayResults(data) {
        emptyState.classList.add('hidden');
        resultDisplay.classList.remove('hidden');

        statLines.textContent = data.total_lines || 0;
        statWords.textContent = data.total_words || 0;
        tamilEditor.value = data.full_text || '';

        // Line-by-line breakdown
        linesBreakdownList.innerHTML = '';
        if (data.lines && data.lines.length > 0) {
            data.lines.forEach((line) => {
                const item = document.createElement('div');
                item.className = 'history-item';
                item.style.padding = '0.5rem';
                item.style.borderBottom = '1px solid #e2e8f0';
                item.innerHTML = `
                    <div style="font-size:0.8rem;color:#64748b;font-weight:600;">Line ${line.line_idx + 1} (${line.words.length} words):</div>
                    <div style="font-family:'Noto Sans Tamil',sans-serif;font-size:1rem;color:#0f172a;margin-top:2px;">${line.line_text}</div>
                `;
                linesBreakdownList.appendChild(item);
            });
        }
    }

    function setLoading(isLoading) {
        recognizeDocBtn.disabled = isLoading;
        if (isLoading) {
            uploadSpinner.classList.remove('hidden');
            recognizeDocBtn.querySelector('.btn-label').textContent = 'Processing OCR...';
        } else {
            uploadSpinner.classList.add('hidden');
            recognizeDocBtn.querySelector('.btn-label').textContent = 'Extract & Recognize Tamil Text';
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.classList.remove('hidden');
    }

    function hideError() {
        errorAlert.classList.add('hidden');
    }

    // Copy to clipboard
    copyBtn.addEventListener('click', () => {
        const text = tamilEditor.value;
        if (!text) return;
        navigator.clipboard.writeText(text).then(() => {
            copyLabel.textContent = 'Copied!';
            setTimeout(() => { copyLabel.textContent = 'Copy Unicode Text'; }, 2000);
        }).catch(() => {
            tamilEditor.select();
            document.execCommand('copy');
            copyLabel.textContent = 'Copied!';
            setTimeout(() => { copyLabel.textContent = 'Copy Unicode Text'; }, 2000);
        });
    });

    // Exports
    async function triggerExport(endpoint, filename, mimetype) {
        const text = tamilEditor.value;
        if (!text) {
            showError('No text available to export.');
            return;
        }

        try {
            const resp = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!resp.ok) throw new Error('Export request failed.');
            const blob = await resp.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (err) {
            showError('Export failed: ' + err.message);
        }
    }

    exportTxtBtn.addEventListener('click', () => {
        triggerExport('/api/export/txt', 'tamil_document_text.txt', 'text/plain');
    });

    exportDocxBtn.addEventListener('click', () => {
        triggerExport('/api/export/docx', 'tamil_document_text.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
    });

    exportPdfBtn.addEventListener('click', () => {
        triggerExport('/api/export/pdf', 'tamil_document_text.pdf', 'application/pdf');
    });
});
