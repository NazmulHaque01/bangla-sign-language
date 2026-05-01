// Utility functions for web deployment

class Utils {
    /**
     * Extract keypoints from hand landmarks
     */
    /**
 * Extract and normalize keypoints (Hands + Upper Body Pose)
 */
    static extractKeypoints(handLandmarks, poseLandmarksArray) {
        if (!handLandmarks || handLandmarks.length === 0) {
            return null;
        }

        let handKps = [];
        let wristCoords = null;

        // 1. Process hands (126 features)
        for (let handIdx = 0; handIdx < Math.min(2, handLandmarks.length); handIdx++) {
            const landmarks = handLandmarks[handIdx];
            if (handIdx === 0) {
                wristCoords = [landmarks[0].x, landmarks[0].y, landmarks[0].z];
            }
            for (let landmark of landmarks) {
                handKps.push(landmark.x, landmark.y, landmark.z);
            }
        }
        while (handKps.length < 126) handKps.push(0.0);

        // 2. Process pose (27 features)
        let poseKps = [];
        let noseCoords = null;

        if (poseLandmarksArray && poseLandmarksArray.length > 0) {
            const poseLandmarks = poseLandmarksArray[0]; // First person
            const nose = poseLandmarks[0];
            noseCoords = [nose.x, nose.y, nose.z];

            for (const idx of CONFIG.FEATURES.targetPoseIndices) {
                const lm = poseLandmarks[idx];
                poseKps.push(lm.x, lm.y, lm.z);
            }
        } else {
            for (let i = 0; i < 27; i++) poseKps.push(0.0);
        }

        // 3. Combine to 153 features
        let combined = handKps.concat(poseKps);

        // 4. RELATIVE NORMALIZATION (Center on Nose, fallback to Wrist)
        const refPoint = noseCoords || wristCoords || [0.0, 0.0, 0.0];

        for (let i = 0; i < combined.length; i += 3) {
            // Only shift active (non-zero padding) coordinates
            if (combined[i] !== 0 || combined[i + 1] !== 0 || combined[i + 2] !== 0) {
                combined[i] -= refPoint[0];
                combined[i + 1] -= refPoint[1];
                combined[i + 2] -= refPoint[2];
            }
        }

        // Final safety padding
        while (combined.length < CONFIG.FEATURES.totalFeatures) {
            combined.push(0.0);
        }

        return combined.slice(0, CONFIG.FEATURES.totalFeatures);
    }

    /**
     * Normalize keypoints using scaler parameters (StandardScaler)
     */
    static normalizeKeypoints(keypoints, scaler) {
        if (!keypoints || !scaler) return null;

        const normalized = new Float32Array(keypoints.length);
        for (let i = 0; i < keypoints.length; i++) {
            const mean = scaler.mean[i];
            const scale = scaler.scale[i] || Math.sqrt(scaler.var[i] + 1e-7);
            normalized[i] = (keypoints[i] - mean) / scale;
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