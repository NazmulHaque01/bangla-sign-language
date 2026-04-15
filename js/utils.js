// Utility functions for web deployment

class Utils {
    /**
     * Extract keypoints from hand landmarks
     */
    static extractKeypoints(handLandmarks) {
        if (!handLandmarks || handLandmarks.length === 0) {
            return null;
        }

        let keypoints = [];

        // Extract up to 2 hands
        // handLandmarks[i] IS the array of 21 NormalizedLandmark objects
        for (let handIdx = 0; handIdx < Math.min(2, handLandmarks.length); handIdx++) {
            const landmarks = handLandmarks[handIdx];
            for (let landmark of landmarks) {
                keypoints.push(landmark.x);
                keypoints.push(landmark.y);
                keypoints.push(landmark.z);
            }
        }

        // Pad to required size (126 = 21 landmarks * 3 coords * 2 hands)
        while (keypoints.length < CONFIG.FEATURES.totalFeatures) {
            keypoints.push(0);
        }

        return keypoints.slice(0, CONFIG.FEATURES.totalFeatures);
    }

    /**
     * Normalize keypoints using scaler parameters
     */
    static normalizeKeypoints(keypoints, scaler) {
        if (!keypoints || !scaler) return null;

        const normalized = [];
        for (let i = 0; i < keypoints.length; i++) {
            const mean = scaler.mean[i];
            const variance = scaler.var[i];
            normalized.push((keypoints[i] - mean) / Math.sqrt(variance + 1e-7));
        }
        return normalized;
    }

    /**
     * Format timestamp
     */
    static formatTime() {
        const now = new Date();
        return now.toLocaleTimeString();
    }

    /**
     * Format confidence as percentage
     */
    static formatConfidence(conf) {
        return (conf * 100).toFixed(1) + '%';
    }

    /**
     * Get top N predictions
     */
    static getTopPredictions(predictions, labels, n = 3) {
        const predsArray = Array.from(predictions);
        const indexed = predsArray.map((val, idx) => ({
            index: idx,
            label: labels[idx.toString()] || 'Unknown',
            confidence: val
        }));

        return indexed.sort((a, b) => b.confidence - a.confidence).slice(0, n);
    }

    /**
     * Update UI element safely
     */
    static updateElement(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = content;
        }
    }

    /**
     * Show/hide element
     */
    static toggleElement(elementId, show) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = show ? 'block' : 'none';
        }
    }

    /**
     * Add CSS class
     */
    static addClass(elementId, className) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.add(className);
        }
    }

    /**
     * Remove CSS class
     */
    static removeClass(elementId, className) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.remove(className);
        }
    }

    /**
     * Get device camera constraints
     */
    static getCameraConstraints() {
        return {
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        };
    }

    /**
     * Log message with timestamp
     */
    static log(message, type = 'info') {
        const timestamp = this.formatTime();
        const prefix = {
            'info': '📘',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️'
        }[type] || '📌';

        console.log(`${prefix} [${timestamp}] ${message}`);
    }
}