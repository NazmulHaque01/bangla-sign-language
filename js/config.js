// Configuration file for web deployment

const CONFIG = {
    // Model paths
    MODEL_PATH: './models/model_weights.json',
    SCALER_PATH: './data/scaler.json',
    LABELS_PATH: './data/class_labels.json',
    
    // Hand Landmarker settings
    HAND_LANDMARKER: {
        numHands: 2,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    },
    
    // Inference settings
    INFERENCE: {
        minConfidence: 0.3,
        updateFrequency: 1  // Update every N frames
    },
    
    // UI settings
    UI: {
        maxHistoryItems: 15,
        fpsUpdateInterval: 500  // ms
    },
    
    // Feature extraction
    FEATURES: {
        landmarksPerHand: 21,
        coordsPerLandmark: 3,  // x, y, z
        maxHands: 2,
        totalFeatures: 126  // 21 * 3 * 2
    }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}