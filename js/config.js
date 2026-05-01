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

    // Pose Landmarker settings (NEW)
    POSE_LANDMARKER: {
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    },

    // Inference settings
    INFERENCE: {
        minConfidence: 0.3,
        updateFrequency: 1,  // Update every N frames
        sustainedDuration: 1000, // ms (1 second)
        sustainedConfidence: 0.86 // Average confidence required to commit a sign
    },

    // UI settings
    UI: {
        maxHistoryItems: 15,
        fpsUpdateInterval: 500  // ms
    },

    // Feature extraction (UPDATED)
    FEATURES: {
        landmarksPerHand: 21,
        coordsPerLandmark: 3,  // x, y, z
        maxHands: 2,
        targetPoseIndices: [0, 2, 5, 7, 8, 11, 12, 13, 14], // Nose, Eyes, Ears, Shoulders, Elbows
        totalFeatures: 153  // 126 (Hands) + 27 (Pose)
    }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}