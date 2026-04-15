// Main inference logic for web deployment
// Uses pure JavaScript neural network inference (no WASM dependency)

let model = null;
let scaler = null;
let labels = null;
let handLandmarker = null;
let video = null;
let isRunning = false;
let lastVideoTime = -1;

// Statistics
let stats = {
    frameCount: 0,
    detections: 0,
    confidences: [],
    inferenceTimes: [],
    startTime: 0,
    lastUpdateTime: 0
};

// ============================================================================
// PURE JS NEURAL NETWORK INFERENCE
// ============================================================================

/**
 * Simple MLP inference engine.
 * Supports Dense layers with ReLU and Softmax activations.
 */
class SimpleModel {
    constructor(modelData) {
        this.layers = modelData.layers;
        this.inputSize = modelData.input_size;
        this.outputSize = modelData.output_size;
    }

    /**
     * Run forward pass through the network
     * @param {Float32Array|number[]} input - Input vector
     * @returns {Float32Array} Output probabilities
     */
    predict(input) {
        let x = input instanceof Float32Array ? input : new Float32Array(input);

        for (const layer of this.layers) {
            x = this._denseLayer(x, layer.weights, layer.bias, layer.activation);
        }

        return x;
    }

    /**
     * Dense (fully connected) layer: output = activation(weights @ input + bias)
     */
    _denseLayer(input, weights, bias, activation) {
        const outSize = weights.length;
        const output = new Float32Array(outSize);

        for (let i = 0; i < outSize; i++) {
            let sum = bias[i];
            const row = weights[i];
            for (let j = 0; j < row.length; j++) {
                sum += row[j] * input[j];
            }
            output[i] = sum;
        }

        // Apply activation
        if (activation === 'relu') {
            for (let i = 0; i < output.length; i++) {
                if (output[i] < 0) output[i] = 0;
            }
        } else if (activation === 'softmax') {
            let max = -Infinity;
            for (let i = 0; i < output.length; i++) {
                if (output[i] > max) max = output[i];
            }
            let sum = 0;
            for (let i = 0; i < output.length; i++) {
                output[i] = Math.exp(output[i] - max);
                sum += output[i];
            }
            for (let i = 0; i < output.length; i++) {
                output[i] /= sum;
            }
        }

        return output;
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Wait for MediaPipe ES module to expose globals
 */
function waitForMediaPipe(timeout = 30000) {
    return new Promise((resolve, reject) => {
        if (window.FilesetResolver && window.HandLandmarker) {
            resolve();
            return;
        }
        const startTime = Date.now();
        const check = () => {
            if (window.FilesetResolver && window.HandLandmarker) {
                resolve();
            } else if (Date.now() - startTime > timeout) {
                reject(new Error('MediaPipe library failed to load (timeout)'));
            } else {
                setTimeout(check, 100);
            }
        };
        check();
    });
}

/**
 * Load all models and initialize on page load
 */
async function initializeApp() {
    try {
        Utils.log('Starting initialization...', 'info');
        updateStatus('Loading model...', 'loading');

        // Wait for MediaPipe ES module to be ready
        Utils.log('Waiting for MediaPipe...', 'info');
        await waitForMediaPipe();
        Utils.log('MediaPipe loaded', 'success');

        // Load model weights (pure JS, no WASM needed)
        await loadModel();

        // Load scaler
        await loadScaler();

        // Load labels
        await loadLabels();

        // Initialize MediaPipe Hand Landmarker
        await initializeHandLandmarker();

        Utils.log('Initialization complete!', 'success');
        updateStatus('Ready! Click "Start Camera" to begin', 'ready');

    } catch (error) {
        Utils.log(`Initialization error: ${error.message}`, 'error');
        console.error('Full initialization error:', error);
        updateStatus(`Error: ${error.message}`, 'error');
    }
}

/**
 * Load model weights from JSON
 */
async function loadModel() {
    try {
        Utils.log('Loading model weights...', 'info');
        const response = await fetch(CONFIG.MODEL_PATH);
        if (!response.ok) throw new Error('Model weights file not found');
        const modelData = await response.json();
        model = new SimpleModel(modelData);
        Utils.log(`Model loaded (${modelData.layers.length} layers)`, 'success');
    } catch (error) {
        throw new Error(`Failed to load model: ${error.message}`);
    }
}

/**
 * Load feature scaler
 */
async function loadScaler() {
    try {
        Utils.log('Loading scaler...', 'info');
        const response = await fetch(CONFIG.SCALER_PATH);
        if (!response.ok) throw new Error('Scaler file not found');
        scaler = await response.json();
        Utils.log('Scaler loaded', 'success');
    } catch (error) {
        throw new Error(`Failed to load scaler: ${error.message}`);
    }
}

/**
 * Load class labels
 */
async function loadLabels() {
    try {
        Utils.log('Loading class labels...', 'info');
        const response = await fetch(CONFIG.LABELS_PATH);
        if (!response.ok) throw new Error('Labels file not found');
        labels = await response.json();
        Utils.log(`Loaded ${Object.keys(labels).length} gesture classes`, 'success');
    } catch (error) {
        throw new Error(`Failed to load labels: ${error.message}`);
    }
}

/**
 * Initialize MediaPipe Hand Landmarker
 */
async function initializeHandLandmarker() {
    try {
        Utils.log('Initializing MediaPipe Hand Landmarker...', 'info');

        // FilesetResolver and HandLandmarker are exposed on window by
        // the ES module script in index.html
        const vision = await window.FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );

        handLandmarker = await window.HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: CONFIG.HAND_LANDMARKER.modelAssetPath,
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numHands: CONFIG.HAND_LANDMARKER.numHands,
            minHandDetectionConfidence: CONFIG.HAND_LANDMARKER.minDetectionConfidence,
            minTrackingConfidence: CONFIG.HAND_LANDMARKER.minTrackingConfidence
        });

        Utils.log('Hand Landmarker initialized', 'success');
    } catch (error) {
        console.error('Hand Landmarker init error:', error);
        throw new Error(`Failed to initialize Hand Landmarker: ${error.message}`);
    }
}

// ============================================================================
// CAMERA CONTROL
// ============================================================================

/**
 * Start webcam stream
 */
async function startCamera() {
    try {
        Utils.log('Starting camera...', 'info');

        const constraints = Utils.getCameraConstraints();
        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        video = document.getElementById('video');
        video.srcObject = stream;

        // Wait for video to be ready
        video.onloadedmetadata = () => {
            video.play();
            isRunning = true;
            lastVideoTime = -1;
            stats.startTime = Date.now();
            stats.frameCount = 0;

            // Update UI
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;

            updateStatus('Camera active. Show your hand gestures!', 'ready');
            Utils.log('Camera started', 'success');

            // Start inference loop
            requestAnimationFrame(inferenceLoop);
        };

    } catch (error) {
        Utils.log(`Camera error: ${error.message}`, 'error');
        updateStatus(`Camera error: ${error.message}`, 'error');
    }
}

/**
 * Stop webcam stream
 */
function stopCamera() {
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }

    isRunning = false;

    // Update UI
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;

    updateStatus('Camera stopped', 'ready');
    Utils.log('Camera stopped', 'info');
}

// ============================================================================
// INFERENCE
// ============================================================================

/**
 * Main inference loop
 */
async function inferenceLoop() {
    if (!isRunning || !handLandmarker || !video) return;

    stats.frameCount++;

    try {
        const startTime = performance.now();

        // Only run detection when we have a new video frame
        const currentTime = video.currentTime;
        if (currentTime !== lastVideoTime) {
            lastVideoTime = currentTime;

            // Detect hands
            const results = handLandmarker.detectForVideo(video, performance.now());

            // Extract keypoints
            let keypoints = null;
            if (results.landmarks && results.landmarks.length > 0) {
                keypoints = Utils.extractKeypoints(results.landmarks);
            }

            // Predict gesture
            if (keypoints) {
                const prediction = predictGesture(keypoints);

                if (prediction && prediction.confidence >= CONFIG.INFERENCE.minConfidence) {
                    stats.detections++;
                    stats.confidences.push(prediction.confidence);

                    const inferenceTime = performance.now() - startTime;
                    stats.inferenceTimes.push(inferenceTime);

                    // Update UI
                    updatePredictionDisplay(prediction);
                    addToHistory(prediction.gesture, prediction.confidence);
                    updateTopPredictions(prediction.allPredictions);

                    // Log
                    Utils.log(`Detected: ${prediction.gesture} (${Utils.formatConfidence(prediction.confidence)})`, 'success');
                }
            }
        }

        // Update statistics periodically
        if (Date.now() - stats.lastUpdateTime > CONFIG.UI.fpsUpdateInterval) {
            updateStatistics();
            stats.lastUpdateTime = Date.now();
        }

    } catch (error) {
        Utils.log(`Inference error: ${error.message}`, 'error');
        console.error('Full inference error:', error);
    }

    if (isRunning) {
        requestAnimationFrame(inferenceLoop);
    }
}

/**
 * Predict gesture from keypoints using pure JS model
 */
function predictGesture(keypoints) {
    if (!keypoints || !model || !scaler) return null;

    try {
        // Normalize keypoints
        const normalized = Utils.normalizeKeypoints(keypoints, scaler);
        if (!normalized) return null;

        // Run inference (pure JS, no TF.js tensor needed)
        const predArray = model.predict(normalized);

        // Find max confidence
        let maxIdx = 0;
        let maxConf = predArray[0];

        for (let i = 1; i < predArray.length; i++) {
            if (predArray[i] > maxConf) {
                maxConf = predArray[i];
                maxIdx = i;
            }
        }

        const gestureName = labels[maxIdx.toString()] || 'Unknown';

        return {
            gesture: gestureName,
            confidence: maxConf,
            allPredictions: Array.from(predArray),
            index: maxIdx
        };

    } catch (error) {
        Utils.log(`Prediction error: ${error.message}`, 'error');
        console.error('Full prediction error:', error);
        return null;
    }
}

// ============================================================================
// UI UPDATE FUNCTIONS
// ============================================================================

/**
 * Update prediction display
 */
function updatePredictionDisplay(prediction) {
    const gestureName = document.getElementById('gestureName');
    const confidence = document.getElementById('confidenceText');
    const confidenceFill = document.getElementById('confidenceFill');

    gestureName.textContent = prediction.gesture;
    confidence.textContent = `Confidence: ${Utils.formatConfidence(prediction.confidence)}`;
    confidenceFill.style.width = (prediction.confidence * 100) + '%';

    // Color based on confidence
    if (prediction.confidence > 0.8) {
        confidenceFill.style.background = 'linear-gradient(90deg, #48bb78, #38a169)';
    } else if (prediction.confidence > 0.6) {
        confidenceFill.style.background = 'linear-gradient(90deg, #ed8936, #dd6b20)';
    } else {
        confidenceFill.style.background = 'linear-gradient(90deg, #f56565, #e53e3e)';
    }
}

/**
 * Add detection to history
 */
function addToHistory(gesture, confidence) {
    const history = document.getElementById('history');

    // Remove empty message if exists
    const emptyMsg = history.querySelector('.history-empty');
    if (emptyMsg) {
        emptyMsg.remove();
    }

    // Create new item
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
        <span class="gesture">${gesture}</span>
        <span class="confidence">${Utils.formatConfidence(confidence)}</span>
        <span class="time">${Utils.formatTime()}</span>
    `;

    history.insertBefore(item, history.firstChild);

    // Keep only last N items
    while (history.children.length > CONFIG.UI.maxHistoryItems) {
        history.removeChild(history.lastChild);
    }
}

/**
 * Update top predictions
 */
function updateTopPredictions(allPredictions) {
    const topContainer = document.getElementById('topPredictions');
    const topPreds = Utils.getTopPredictions(allPredictions, labels, 3);

    topContainer.innerHTML = '';

    topPreds.forEach((pred, idx) => {
        const item = document.createElement('div');
        item.className = 'top-pred-item';
        item.innerHTML = `
            <span>${idx + 1}. ${pred.label}</span>
            <div class="top-pred-bar">
                <div class="top-pred-fill" style="width: ${pred.confidence * 100}%"></div>
            </div>
            <span>${Utils.formatConfidence(pred.confidence)}</span>
        `;
        topContainer.appendChild(item);
    });
}

/**
 * Update statistics
 */
function updateStatistics() {
    const elapsed = (Date.now() - stats.startTime) / 1000;
    const fps = stats.frameCount / elapsed;

    Utils.updateElement('fpsCount', fps.toFixed(1));
    Utils.updateElement('detections', stats.detections);

    if (stats.confidences.length > 0) {
        const avgConf = stats.confidences.reduce((a, b) => a + b) / stats.confidences.length;
        Utils.updateElement('avgConfidence', (avgConf * 100).toFixed(0) + '%');
    }

    if (stats.inferenceTimes.length > 0) {
        const avgInference = stats.inferenceTimes.reduce((a, b) => a + b) / stats.inferenceTimes.length;
        Utils.updateElement('inference', avgInference.toFixed(1) + 'ms');
    }
}

/**
 * Update status bar
 */
function updateStatus(message, type = 'info') {
    const status = document.getElementById('status');
    const statusText = document.getElementById('statusText');
    const statusIcon = document.getElementById('statusIcon');

    statusText.textContent = message;

    // Update classes
    status.className = 'status';
    status.classList.add(`status-${type}`);

    // Update icon
    const icons = {
        'loading': '⏳',
        'ready': '✅',
        'error': '❌',
        'info': 'ℹ️'
    };
    statusIcon.textContent = icons[type] || '🔔';
}

// ============================================================================
// INITIALIZATION ON PAGE LOAD
// ============================================================================

// Start initialization when DOM is ready
// The MediaPipe module loads async, so initializeApp handles waiting for it
if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
window.addEventListener('beforeunload', stopCamera);