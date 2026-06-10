document.addEventListener('DOMContentLoaded', () => {
    // ── Element references (match index.html IDs exactly) ─────────────────────
    const dropzone        = document.getElementById('dropzone');
    const fileInput       = document.getElementById('file-input');
    const fileInfoBar     = document.getElementById('file-info-bar');
    const fileNameEl      = document.getElementById('file-name');
    const fileSizeEl      = document.getElementById('file-size');
    const uploadBtn       = document.getElementById('upload-btn');
    const resetBtn        = document.getElementById('reset-btn');
    const loadingPanel    = document.getElementById('loading-panel');
    const loadingMessage  = document.getElementById('loading-message');
    const loadingSubmsg   = document.getElementById('loading-submessage');
    const dashboardGrid   = document.getElementById('dashboard-grid');
    const analyticsSection= document.getElementById('analytics-section');
    const mediaViewer     = document.getElementById('media-viewer');
    const downloadMediaBtn= document.getElementById('download-media-btn');
    const downloadCsvBtn  = document.getElementById('download-csv-btn');
    const errorAlert      = document.getElementById('error-alert');
    const errorMessage    = document.getElementById('error-message');
    const confSlider      = document.getElementById('conf-threshold');
    const confVal         = document.getElementById('conf-val');
    const totalDetVal     = document.getElementById('total-detections-val');
    const peakDetVal      = document.getElementById('peak-detections-val');
    const classCountGrid  = document.getElementById('class-count-grid');
    const tableBody       = document.getElementById('detections-table-body');

    let selectedFile = null;
    let chartInstance = null;

    // ── Initial UI state ───────────────────────────────────────────────────────
    dashboardGrid.style.display    = 'none';
    analyticsSection.style.display = 'none';
    loadingPanel.style.display     = 'none';
    errorAlert.style.display       = 'none';

    // ── Confidence slider live display ─────────────────────────────────────────
    confSlider.addEventListener('input', () => {
        confVal.textContent = parseFloat(confSlider.value).toFixed(2);
    });

    // ── Click on dropzone → open file dialog ──────────────────────────────────
    dropzone.addEventListener('click', () => fileInput.click());

    // ── Drag-and-drop handlers ────────────────────────────────────────────────
    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) setFile(files[0]);
    });

    // ── File input change (click-to-browse) ───────────────────────────────────
    fileInput.addEventListener('change', e => {
        if (e.target.files.length > 0) setFile(e.target.files[0]);
    });

    // ── Set selected file + show info bar ─────────────────────────────────────
    function setFile(file) {
        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/');
        if (!isImage && !isVideo) {
            showError('Invalid file format. Please upload an image or video (JPG, PNG, MP4, AVI, MOV).');
            return;
        }
        selectedFile = file;
        fileNameEl.textContent = file.name;
        fileSizeEl.textContent = formatSize(file.size);
        fileInfoBar.style.display = 'flex';
        uploadBtn.classList.remove('btn-disabled');
        errorAlert.style.display = 'none';
    }

    // ── Reset / Clear ─────────────────────────────────────────────────────────
    resetBtn.addEventListener('click', resetUI);

    function resetUI() {
        selectedFile = null;
        fileInput.value = '';
        fileInfoBar.style.display  = 'none';
        uploadBtn.classList.add('btn-disabled');
        dashboardGrid.style.display    = 'none';
        analyticsSection.style.display = 'none';
        errorAlert.style.display       = 'none';
        mediaViewer.innerHTML = '';
        tableBody.innerHTML   = '';
        classCountGrid.innerHTML = '';
        totalDetVal.textContent  = '--';
        peakDetVal.textContent   = '--';
        if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
    }

    // ── Start Processing button ───────────────────────────────────────────────
    uploadBtn.addEventListener('click', () => {
        if (!selectedFile || uploadBtn.classList.contains('btn-disabled')) return;
        startUpload();
    });

    function startUpload() {
        // Build form data — include the user's confidence threshold
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('conf', confSlider.value);

        // UI: show spinner
        loadingPanel.style.display     = 'flex';
        dashboardGrid.style.display    = 'none';
        analyticsSection.style.display = 'none';
        errorAlert.style.display       = 'none';
        setLoadingState('Uploading media file...', 'Sending data to Flask server');

        fetch('/upload', { method: 'POST', body: formData })
            .then(res => {
                setLoadingState('Running YOLO inference...', 'Detecting vehicles frame-by-frame');
                if (!res.ok) return res.json().then(err => { throw new Error(err.error || 'Server error'); });
                return res.json();
            })
            .then(data => {
                loadingPanel.style.display = 'none';
                renderResult(data);
            })
            .catch(err => {
                loadingPanel.style.display = 'none';
                showError(err.message);
            });
    }

    function setLoadingState(msg, sub) {
        loadingMessage.textContent = msg;
        loadingSubmsg.textContent  = sub;
    }

    // ── Render result media + analytics ──────────────────────────────────────
    function renderResult(data) {
        mediaViewer.innerHTML = '';

        if (data.type === 'video') {
            const video = document.createElement('video');
            video.className  = 'result-media';
            video.controls   = true;
            video.autoplay   = true;
            video.loop       = true;
            video.muted      = true;
            video.playsInline = true;
            const source     = document.createElement('source');
            source.src       = data.url + '?t=' + Date.now();
            source.type      = 'video/mp4';
            video.appendChild(source);
            mediaViewer.appendChild(video);
            video.load();
        } else {
            const img    = document.createElement('img');
            img.src      = data.url;
            img.className = 'result-media';
            img.alt      = 'Annotated detection result';
            mediaViewer.appendChild(img);
        }

        downloadMediaBtn.href = data.url;
        downloadCsvBtn.href   = data.csv_url;

        dashboardGrid.style.display    = 'grid';
        analyticsSection.style.display = 'flex';

        // Load CSV for analytics
        fetch(data.csv_url)
            .then(r => r.text())
            .then(csv => buildAnalytics(csv))
            .catch(() => { /* CSV optional — analytics stay blank */ });
    }

    // ── Analytics: parse CSV → stats + chart + table ─────────────────────────
    function buildAnalytics(csvText) {
        const lines = csvText.trim().split('\n');
        if (lines.length < 2) return;

        const rows = lines.slice(1).map(l => {
            const [frame_id, class_name, confidence, x1, y1, x2, y2] = l.split(',');
            return { frame_id: +frame_id, class_name, confidence: +confidence,
                     x1: +x1, y1: +y1, x2: +x2, y2: +y2 };
        });

        // ── Summary stats ──────────────────────────────────────────────────────
        const classCounts = {};
        const frameCounts = {};
        rows.forEach(r => {
            classCounts[r.class_name] = (classCounts[r.class_name] || 0) + 1;
            frameCounts[r.frame_id]   = (frameCounts[r.frame_id]   || 0) + 1;
        });

        const totalInstances = rows.length;
        const peakPerFrame   = Object.values(frameCounts).length
            ? Math.max(...Object.values(frameCounts)) : 0;

        totalDetVal.textContent = totalInstances.toLocaleString();
        peakDetVal.textContent  = peakPerFrame;

        // ── Class breakdown chips ──────────────────────────────────────────────
        const CLASS_COLOR_HEX = {
            'Ambulance':  '#ff8c00',
            'Bus':        '#5b8dee',
            'Car':        '#22c55e',
            'Motorcycle': '#facc15',
            'Truck':      '#ef4444',
        };
        classCountGrid.innerHTML = '';
        Object.entries(classCounts)
            .sort((a, b) => b[1] - a[1])
            .forEach(([cls, cnt]) => {
                const color = CLASS_COLOR_HEX[cls] || '#8b5cf6';
                const chip  = document.createElement('div');
                chip.className = 'class-chip';
                chip.style.cssText = `border-left: 3px solid ${color};`;
                chip.innerHTML = `<span class="chip-label">${cls}</span><span class="chip-count" style="color:${color}">${cnt}</span>`;
                classCountGrid.appendChild(chip);
            });

        // ── Chart: traffic density by frame ───────────────────────────────────
        const frameIds = Object.keys(frameCounts).map(Number).sort((a,b)=>a-b);
        const densities = frameIds.map(f => frameCounts[f]);

        // Build per-class stacked data
        const allClasses = Object.keys(classCounts);
        const classFrameData = {};
        allClasses.forEach(cls => {
            classFrameData[cls] = frameIds.map(fid => {
                return rows.filter(r => r.frame_id === fid && r.class_name === cls).length;
            });
        });

        if (chartInstance) chartInstance.destroy();
        const ctx = document.getElementById('analysis-chart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: frameIds.length > 200
                    ? frameIds.filter((_, i) => i % Math.ceil(frameIds.length / 100) === 0)
                    : frameIds,
                datasets: allClasses.map(cls => ({
                    label: cls,
                    data: frameIds.length > 200
                        ? classFrameData[cls].filter((_, i) => i % Math.ceil(frameIds.length / 100) === 0)
                        : classFrameData[cls],
                    backgroundColor: (CLASS_COLOR_HEX[cls] || '#8b5cf6') + 'cc',
                    borderWidth: 0,
                    borderRadius: 2,
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#cbd5e1', font: { size: 11 } } },
                    tooltip: { backgroundColor: 'rgba(15,23,42,0.9)', titleColor: '#f1f5f9', bodyColor: '#cbd5e1' }
                },
                scales: {
                    x: { stacked: true, ticks: { color: '#64748b', maxTicksLimit: 15 }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { stacked: true, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.07)' }, beginAtZero: true }
                }
            }
        });

        // ── Table: show last 200 rows ──────────────────────────────────────────
        tableBody.innerHTML = '';
        const displayRows = rows.slice(-200);
        displayRows.forEach(r => {
            const tr = document.createElement('tr');
            const w  = r.x2 - r.x1, h = r.y2 - r.y1;
            const color = CLASS_COLOR_HEX[r.class_name] || '#8b5cf6';
            tr.innerHTML = `
                <td>${r.frame_id}</td>
                <td><span style="color:${color};font-weight:600;">${r.class_name}</span></td>
                <td>${(r.confidence * 100).toFixed(1)}%</td>
                <td>(${r.x1}, ${r.y1})</td>
                <td>(${r.x2}, ${r.y2})</td>
                <td>${w} × ${h}</td>`;
            tableBody.appendChild(tr);
        });
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    function formatSize(bytes) {
        if (bytes < 1024)       return bytes + ' B';
        if (bytes < 1048576)    return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function showError(msg) {
        errorMessage.textContent  = msg;
        errorAlert.style.display  = 'flex';
        setTimeout(() => errorAlert.style.display = 'none', 7000);
    }
});
