# How to Deploy a New Model

## Step 1: Replace these 3 files

| File | Location | What to replace |
|------|----------|-----------------|
| `sign_language_keypoint_model.tflite` | `models/` | Your new `.tflite` model |
| `class_labels.json` | `data/` | Your new class labels |
| `scaler.json` | `data/` | Your new scaler |

## Step 2: Generate model_weights.json

Open CMD in the project folder and run:

```
python extract_model.py
```

This auto-detects the architecture and creates `models/model_weights.json`.
Verify the output says **"Match: True"** at the end.

## Step 3: Test

```
python -m http.server 8000
```

Open http://localhost:8000 — status should show **"Ready!"** with a green checkmark.

---

**That's it.** No code changes needed unless your model uses a different input format (not 21 hand landmarks × 3 coords × 2 hands = 126 features).
