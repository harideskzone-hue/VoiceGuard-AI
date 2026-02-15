document.addEventListener('DOMContentLoaded', () => {
    // ===== DOM Elements =====
    const recordBtn = document.getElementById('recordBtn');
    const recordLabel = document.getElementById('recordLabel');
    const fileInput = document.getElementById('fileInput');
    const canvas = document.getElementById('visualizer');
    const vizOverlay = document.getElementById('vizOverlay');
    const dropZone = document.getElementById('dropZone');

    // States
    const awaitingState = document.getElementById('awaitingState');
    const analyzingState = document.getElementById('analyzingState');
    const resultState = document.getElementById('resultState');

    // Result elements
    const verdictCard = document.getElementById('verdictCard');
    const verdictIcon = document.getElementById('verdictIcon');
    const classificationEl = document.getElementById('classification');
    const confidenceScoreEl = document.getElementById('confidenceScore');
    const explanationText = document.getElementById('explanationText');
    const ringProgress = document.getElementById('ringProgress');
    const methodBadge = document.getElementById('methodBadge');

    // Pipeline Detail elements
    const sotaBar = document.getElementById('sotaBar');
    const heuristicBar = document.getElementById('heuristicBar');
    const finalBar = document.getElementById('finalBar');
    const sotaScoreText = document.getElementById('sotaScoreText');
    const heuristicScoreText = document.getElementById('heuristicScoreText');
    const finalScoreText = document.getElementById('finalScoreText');
    const modelNameText = document.getElementById('modelNameText');

    // Demo buttons
    const demoHuman = document.getElementById('demoHuman');
    const demoAI = document.getElementById('demoAI');

    // Stats
    const latencyValue = document.getElementById('latencyValue');
    const langValue = document.getElementById('langValue');
    const timeValue = document.getElementById('timeValue');

    // History
    const historyList = document.getElementById('historyList');
    const totalCount = document.getElementById('totalCount');

    // Config
    const API_URL = '/api/voice-detection';
    const API_KEY = 'your-super-secret-api-key-change-this';

    // Audio State
    let audioContext, analyser, mediaRecorder, audioChunks = [];
    let isRecording = false, animationId;
    let analysisCount = 0;

    // ===== Particle Background =====
    const bgCanvas = document.getElementById('bgCanvas');
    const bgCtx = bgCanvas.getContext('2d');
    let particles = [];

    function resizeBgCanvas() {
        bgCanvas.width = window.innerWidth;
        bgCanvas.height = window.innerHeight;
    }

    function initParticles() {
        particles = [];
        const count = Math.floor((bgCanvas.width * bgCanvas.height) / 15000);
        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * bgCanvas.width,
                y: Math.random() * bgCanvas.height,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                r: Math.random() * 1.5 + 0.5,
                a: Math.random() * 0.5 + 0.1
            });
        }
    }

    function drawParticles() {
        bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    bgCtx.beginPath();
                    bgCtx.moveTo(particles[i].x, particles[i].y);
                    bgCtx.lineTo(particles[j].x, particles[j].y);
                    bgCtx.strokeStyle = `rgba(0, 212, 255, ${0.08 * (1 - dist / 120)})`;
                    bgCtx.lineWidth = 0.5;
                    bgCtx.stroke();
                }
            }
        }

        // Draw particles
        for (const p of particles) {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0 || p.x > bgCanvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > bgCanvas.height) p.vy *= -1;

            bgCtx.beginPath();
            bgCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            bgCtx.fillStyle = `rgba(0, 212, 255, ${p.a})`;
            bgCtx.fill();
        }

        requestAnimationFrame(drawParticles);
    }

    window.addEventListener('resize', () => { resizeBgCanvas(); initParticles(); });
    resizeBgCanvas();
    initParticles();
    drawParticles();

    // ===== Canvas Setup =====
    const canvasCtx = canvas.getContext('2d');
    function resizeCanvas() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // ===== Event Listeners =====
    recordBtn.addEventListener('click', toggleRecording);

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) processFile(file);
    });
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) processFile(e.target.files[0]);
    });

    // ===== Recording =====
    async function toggleRecording() {
        if (!isRecording) startRecording();
        else stopRecording();
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000, channelCount: 1,
                    echoCancellation: true, noiseSuppression: true, autoGainControl: true
                }
            });

            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            visualize();

            let mimeType = 'audio/webm;codecs=opus';
            if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) mimeType = 'audio/webm;codecs=opus';
            else if (MediaRecorder.isTypeSupported('audio/webm')) mimeType = 'audio/webm';

            mediaRecorder = new MediaRecorder(stream, { mimeType });
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };

            mediaRecorder.onstop = async () => {
                const blob = new Blob(audioChunks, { type: mimeType });
                const wavBlob = await convertToWav(blob);
                processBlob(wavBlob);
                cancelAnimationFrame(animationId);
                canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
                vizOverlay.classList.remove('hidden');
                if (audioContext && audioContext.state !== 'closed') audioContext.close();
                stream.getTracks().forEach(t => t.stop());
            };

            mediaRecorder.start(100);
            isRecording = true;
            recordBtn.classList.add('recording');
            recordLabel.textContent = 'STOP RECORDING';
            vizOverlay.classList.add('hidden');

        } catch (err) {
            console.error("Mic error:", err);
            alert("Could not access microphone. Please allow permissions.");
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            recordBtn.classList.remove('recording');
            recordLabel.textContent = 'START RECORDING';
        }
    }

    // ===== WAV Conversion =====
    async function convertToWav(blob) {
        try {
            const arrayBuffer = await blob.arrayBuffer();
            const tempCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);
            tempCtx.close();
            const channelData = audioBuffer.getChannelData(0);
            return new Blob([encodeWav(channelData, audioBuffer.sampleRate)], { type: 'audio/wav' });
        } catch (err) {
            console.warn("WAV conversion failed:", err);
            return blob;
        }
    }

    function encodeWav(samples, sampleRate) {
        const bps = 16, ch = 1, ba = ch * (bps / 8);
        const dataSize = samples.length * (bps / 8);
        const buf = new ArrayBuffer(44 + dataSize);
        const v = new DataView(buf);
        const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };

        ws(0, 'RIFF'); v.setUint32(4, 36 + dataSize, true); ws(8, 'WAVE');
        ws(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
        v.setUint16(22, ch, true); v.setUint32(24, sampleRate, true);
        v.setUint32(28, sampleRate * ba, true); v.setUint16(32, ba, true);
        v.setUint16(34, bps, true); ws(36, 'data'); v.setUint32(40, dataSize, true);

        let o = 44;
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            o += 2;
        }
        return buf;
    }

    // ===== Visualizer =====
    function visualize() {
        const bufLen = analyser.frequencyBinCount;
        const data = new Uint8Array(bufLen);

        function draw() {
            animationId = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(data);

            canvasCtx.fillStyle = 'rgba(10, 14, 26, 0.3)';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

            const barW = (canvas.width / bufLen) * 2.5;
            let x = 0;
            for (let i = 0; i < bufLen; i++) {
                const h = data[i] / 2;
                const gradient = canvasCtx.createLinearGradient(0, canvas.height, 0, canvas.height - h);
                gradient.addColorStop(0, 'rgba(0, 212, 255, 0.8)');
                gradient.addColorStop(1, 'rgba(168, 85, 247, 0.6)');
                canvasCtx.fillStyle = gradient;
                canvasCtx.fillRect(x, canvas.height - h, barW, h);
                x += barW + 1;
            }
        }
        draw();
    }

    // ===== File Processing =====
    function processFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            // Get extension from filename (more reliable than MIME type for API)
            // e.g., "audio.mp3" -> "mp3", "audio.mpeg" -> "mp3"
            let format = file.name.split('.').pop().toLowerCase();

            // Map common mismatches if necessary (though extension usually works)
            if (format === 'mpeg') format = 'mp3';

            sendToAPI(e.target.result.split(',')[1], format);
        };
        reader.readAsDataURL(file);
    }

    function processBlob(blob) {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => sendToAPI(reader.result.split(',')[1], 'wav');
    }

    // ===== API Call =====
    async function sendToAPI(base64Audio, format) {
        // Show analyzing state
        awaitingState.classList.add('hidden');
        resultState.classList.add('hidden');
        analyzingState.classList.remove('hidden');

        const startTime = Date.now();
        const lang = document.getElementById('languageSelect').value;

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'x-api-key': API_KEY },
                body: JSON.stringify({ language: lang, audioFormat: format, audioBase64: base64Audio })
            });

            const data = await response.json();
            const latency = Date.now() - startTime;

            if (data.status === 'success') {
                showResult(data, latency, lang);
            } else {
                alert("Error: " + (data.message || "Unknown error"));
                analyzingState.classList.add('hidden');
                awaitingState.classList.remove('hidden');
            }
        } catch (error) {
            console.error("API Error:", error);
            alert("Failed to connect to VoiceGuard API.");
            analyzingState.classList.add('hidden');
            awaitingState.classList.remove('hidden');
        }
    }

    // ===== Show Result =====
    function showResult(data, latency, lang) {
        const isAI = data.classification === 'AI_GENERATED';
        const score = Math.round(data.confidenceScore * 100);

        // Switch states
        analyzingState.classList.add('hidden');
        resultState.classList.remove('hidden');

        // Verdict card styling
        verdictCard.className = 'verdict-card ' + (isAI ? 'ai' : 'human');
        verdictIcon.textContent = isAI ? '🤖' : '✅';
        classificationEl.textContent = isAI ? 'AI GENERATED' : 'HUMAN VERIFIED';

        // Confidence ring animation
        const circumference = 42 * 2 * Math.PI;
        ringProgress.style.strokeDasharray = `${circumference} ${circumference}`;
        ringProgress.style.strokeDashoffset = circumference; // reset
        setTimeout(() => {
            ringProgress.style.strokeDashoffset = circumference - (score / 100) * circumference;
        }, 50);
        confidenceScoreEl.textContent = score;

        // Explanation
        explanationText.textContent = data.explanation || 'Analysis complete.';

        // Method badge
        if (data.details && data.details.method) {
            methodBadge.textContent = data.details.method;
        } else {
            methodBadge.textContent = 'ENSEMBLE';
        }

        // Pipeline detail bars
        if (data.details) {
            const d = data.details;
            const sotaPct = d.sotaScore != null ? (d.sotaScore * 100) : 0;
            const heuristicPct = d.heuristicScore != null ? (d.heuristicScore * 100) : 0;
            const finalPct = data.confidenceScore * 100;

            setTimeout(() => {
                sotaBar.style.width = sotaPct.toFixed(0) + '%';
                heuristicBar.style.width = heuristicPct.toFixed(0) + '%';
                finalBar.style.width = finalPct.toFixed(0) + '%';
            }, 100);

            sotaScoreText.textContent = d.sotaScore != null ? (d.sotaScore * 100).toFixed(1) + '%' : 'N/A';
            heuristicScoreText.textContent = d.heuristicScore != null ? (d.heuristicScore * 100).toFixed(1) + '%' : 'N/A';
            finalScoreText.textContent = finalPct.toFixed(1) + '%';

            modelNameText.textContent = d.modelName ? `Model: ${d.modelName}` : '';
        }

        // Stats
        latencyValue.textContent = latency;
        langValue.textContent = lang;
        const now = new Date();
        timeValue.textContent = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        // Add to history
        addToHistory(isAI, score, lang, now);
    }

    // ===== History =====
    function addToHistory(isAI, score, lang, time) {
        analysisCount++;
        totalCount.textContent = analysisCount;

        // Remove empty state
        const empty = historyList.querySelector('.history-empty');
        if (empty) empty.remove();

        const item = document.createElement('div');
        item.className = `history-item ${isAI ? 'ai' : 'human'}`;
        item.innerHTML = `
            <div class="h-icon">${isAI ? '🤖' : '✅'}</div>
            <div class="h-info">
                <div class="h-label">${isAI ? 'AI Generated' : 'Human Voice'}</div>
                <div class="h-sub">${lang} • ${time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</div>
            </div>
            <div class="h-score" style="color: ${isAI ? 'var(--red)' : 'var(--green)'}">${score}%</div>
        `;

        historyList.insertBefore(item, historyList.firstChild);

        // Keep only last 10
        while (historyList.children.length > 10) {
            historyList.removeChild(historyList.lastChild);
        }
    }

    // ===== Demo Sample Generation =====
    // Generate synthetic audio samples using Web Audio API
    // Human sample: natural speech-like with varied frequencies
    // AI sample: robotic monotone with exact periodicity (what AI sounds like)

    function generateDemoAudio(type) {
        const sampleRate = 16000;
        const duration = 3; // seconds
        const numSamples = sampleRate * duration;
        const samples = new Float32Array(numSamples);

        if (type === 'human') {
            // Simulate natural speech: varying frequencies, amplitude modulation, pauses
            for (let i = 0; i < numSamples; i++) {
                const t = i / sampleRate;
                // Fundamental frequency varies naturally (100-180 Hz)
                const f0 = 140 + 40 * Math.sin(2 * Math.PI * 0.5 * t) + 20 * Math.sin(2 * Math.PI * 1.3 * t);
                // Natural amplitude envelope with "breathing" pauses
                const envPeriod = 0.8 + 0.3 * Math.sin(t);
                const envelope = Math.max(0, Math.sin(2 * Math.PI * t / envPeriod)) * (0.6 + 0.4 * Math.random() * 0.1);
                // Mix harmonics with slight randomness
                const h1 = Math.sin(2 * Math.PI * f0 * t);
                const h2 = 0.5 * Math.sin(2 * Math.PI * 2 * f0 * t + Math.random() * 0.1);
                const h3 = 0.25 * Math.sin(2 * Math.PI * 3 * f0 * t + Math.random() * 0.05);
                // Add slight noise (breathiness)
                const noise = (Math.random() - 0.5) * 0.05;
                samples[i] = envelope * (h1 + h2 + h3) * 0.25 + noise;
            }
        } else {
            // Simulate AI-generated: perfectly periodic, no natural variation
            for (let i = 0; i < numSamples; i++) {
                const t = i / sampleRate;
                // Exact constant frequency (robotic)
                const f0 = 150;
                const envelope = 0.7; // constant amplitude
                // Perfect harmonics with no variation
                const h1 = Math.sin(2 * Math.PI * f0 * t);
                const h2 = 0.5 * Math.sin(2 * Math.PI * 2 * f0 * t);
                const h3 = 0.3 * Math.sin(2 * Math.PI * 3 * f0 * t);
                const h4 = 0.15 * Math.sin(2 * Math.PI * 4 * f0 * t);
                samples[i] = envelope * (h1 + h2 + h3 + h4) * 0.2;
            }
        }

        // Convert to WAV blob
        const wavBuffer = encodeWav(samples, sampleRate);
        return new Blob([wavBuffer], { type: 'audio/wav' });
    }

    demoHuman.addEventListener('click', () => {
        const blob = generateDemoAudio('human');
        processBlob(blob);
    });

    demoAI.addEventListener('click', () => {
        const blob = generateDemoAudio('ai');
        processBlob(blob);
    });
});
