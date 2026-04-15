import pickle
import json
import numpy as np

# Load scaler from Kaggle output
with open('keypoint_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Convert to JSON
scaler_data = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'var': scaler.var_.tolist()
}

# Save
with open('scaler.json', 'w') as f:
    json.dump(scaler_data, f)

print("✅ Scaler converted to JSON")