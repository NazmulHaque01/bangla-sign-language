# Bangla Sign Language Recognition (Web v2)

A real-time Bangla Sign Language (BdSL) recognition system running entirely in the browser using MediaPipe and a custom Pure JavaScript inference engine.

## 🚀 Key Features
- **Pure JS Inference**: No dependency on TensorFlow.js WASM or TFLite runtimes, making it extremely stable across all modern browsers.
- **MediaPipe Integration**: Uses MediaPipe Hand Landmarker and Pose Landmarker for robust keypoint extraction.
- **Hybrid Features**: Utilizes 153 features (126 hand landmarks + 27 upper-body pose landmarks) for high accuracy.
- **Zero Jitter**: Implements a sustained detection logic to ensure signs are only committed after a 1-second hold with high confidence.
- **Normalization**: Relative normalization based on nose/wrist position ensures consistency regardless of user position in frame.

## 📁 Project Structure
- `js/`: Core logic including `inference.js` (Neural Network), `utils.js` (Keypoint processing), and `config.js`.
- `models/`: Contains `model_weights.json` for the pure JS inference engine.
- `data/`: Contains `scaler.json` and class labels.
- `index.html`: Main user interface.
- `css/`: Modern UI styling.

## 🛠️ Local Development
To run this application locally without CORS issues:

1.  **Using Python**:
    ```bash
    python -m http.server 8000
    ```
2.  **Using Node.js**:
    ```bash
    npm install -g http-server
    http-server . -p 8000
    ```
3.  Open `http://localhost:8000` in your browser.

## 🧠 Model Details
The underlying model is a Dense Neural Network (MLP) trained on keypoints. The inference is performed using a custom-built JavaScript matrix multiplication layer to minimize external library overhead and load times.

## 📜 License
This project is for educational and research purposes.
