"""
Extract weights from TFLite model and save as JSON for browser inference.
Handles quantized (int8) weights by dequantizing them.
Auto-detects model architecture - works with any Dense NN model.

Usage: python extract_model.py
Requires: tensorflow, numpy
"""
import json
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='models/sign_language_keypoint_model.tflite')
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_size = int(input_details[0]['shape'][-1])
output_size = int(output_details[0]['shape'][-1])

print(f"Input shape: {input_details[0]['shape']} (features: {input_size})")
print(f"Output shape: {output_details[0]['shape']} (classes: {output_size})")

# Get all tensor details
tensor_details = interpreter.get_tensor_details()

print(f"\nAll tensors ({len(tensor_details)}):")
for t in tensor_details:
    quant = t.get('quantization_parameters', {})
    scales = quant.get('scales', [])
    q_info = "QUANTIZED" if len(scales) > 0 else "float32"
    print(f"  [{t['index']:2d}] shape={str(t['shape']):20s} {q_info:10s} {t['name'][:80]}")


def get_tensor_data(interpreter, index, tensor_details):
    """Get tensor data, dequantizing if necessary."""
    details = None
    for t in tensor_details:
        if t['index'] == index:
            details = t
            break

    tensor = interpreter.get_tensor(index)

    # If quantized, dequantize
    quant = details.get('quantization_parameters', {})
    scales = quant.get('scales', np.array([]))
    zero_points = quant.get('zero_points', np.array([]))

    if len(scales) > 0 and tensor.dtype == np.int8:
        print(f"    Dequantizing tensor {index}: {tensor.shape}")
        # Per-channel quantization: output = (int8_val - zero_point) * scale
        if len(tensor.shape) == 2:
            dequantized = np.zeros_like(tensor, dtype=np.float32)
            for i in range(tensor.shape[0]):
                dequantized[i] = (tensor[i].astype(np.float32) - zero_points[i]) * scales[i]
            return dequantized
        else:
            return ((tensor.astype(np.float32) - zero_points) * scales)

    return tensor


# ============================================================================
# Auto-detect layer structure
# Strategy: find all 2D weight tensors and 1D bias tensors, then match them
# by checking which weight [out, in] has a matching bias [out]
# ============================================================================
print("\n--- Auto-detecting layers ---")

# Separate weight matrices (2D) and bias vectors (1D)
# Skip input/output/intermediate activation tensors
weight_tensors = []
bias_tensors = []

for t in tensor_details:
    shape = tuple(t['shape'])
    name = t['name']
    
    # Skip input tensor
    if t['index'] == input_details[0]['index']:
        continue
    # Skip output tensor
    if t['index'] == output_details[0]['index']:
        continue
    # Skip intermediate activation tensors (batch dimension = 1 and 2D)
    if len(shape) == 2 and shape[0] == 1:
        continue
    
    if len(shape) == 2 and shape[0] != 1:
        weight_tensors.append(t)
    elif len(shape) == 1:
        bias_tensors.append(t)

print(f"  Found {len(weight_tensors)} weight matrices, {len(bias_tensors)} bias vectors")

# Match weights to biases: weight shape [out, in] pairs with bias shape [out]
# Then chain them: layer N output = layer N+1 input
layers_info = []
used_bias = set()

# Sort weights by tracing the network: start from input_size, find matching weight
remaining_weights = list(weight_tensors)
current_in = input_size

while remaining_weights:
    found = False
    for w in remaining_weights:
        w_shape = tuple(w['shape'])
        if w_shape[1] == current_in:
            # Found the next layer weight
            # Find matching bias
            out_size = w_shape[0]
            matching_bias = None
            for b in bias_tensors:
                if b['index'] not in used_bias and tuple(b['shape']) == (out_size,):
                    matching_bias = b
                    break
            
            if matching_bias:
                used_bias.add(matching_bias['index'])
            
            layers_info.append({
                'weight_idx': w['index'],
                'bias_idx': matching_bias['index'] if matching_bias else None,
                'in_size': w_shape[1],
                'out_size': w_shape[0],
                'name': w['name'][:60]
            })
            
            current_in = out_size
            remaining_weights.remove(w)
            found = True
            break
    
    if not found:
        print(f"  WARNING: Could not find layer with input size {current_in}")
        print(f"  Remaining weights: {[(tuple(w['shape']), w['name'][:40]) for w in remaining_weights]}")
        break

print(f"  Detected {len(layers_info)} layers:")
for i, info in enumerate(layers_info):
    is_last = (i == len(layers_info) - 1)
    activation = 'softmax' if is_last else 'relu'
    print(f"    Layer {i+1}: Dense({info['in_size']} -> {info['out_size']}) "
          f"weight_idx={info['weight_idx']}, bias_idx={info['bias_idx']}, "
          f"activation={activation}")

# ============================================================================
# Extract weights
# ============================================================================
print("\n--- Extracting weights ---")
layers = []

for i, info in enumerate(layers_info):
    is_last = (i == len(layers_info) - 1)
    activation = 'softmax' if is_last else 'relu'
    
    print(f"Layer {i+1} ({info['in_size']}->{info['out_size']}, {activation}):")
    w = get_tensor_data(interpreter, info['weight_idx'], tensor_details)
    b = get_tensor_data(interpreter, info['bias_idx'], tensor_details) if info['bias_idx'] is not None else np.zeros(info['out_size'])
    
    layers.append({
        'weights': w.tolist(),
        'bias': b.tolist() if hasattr(b, 'tolist') else b,
        'activation': activation
    })
    print(f"  Weights: {w.shape}, Bias: {np.array(b).shape}")

# Build model JSON
model_data = {
    'input_size': input_size,
    'output_size': output_size,
    'layers': layers
}

# Save as JSON
output_path = 'models/model_weights.json'
json_str = json.dumps(model_data)
with open(output_path, 'w') as f:
    f.write(json_str)

print(f"\nModel saved to {output_path}")
print(f"File size: {len(json_str):,} bytes")
print(f"Architecture: {input_size} -> {' -> '.join(str(l['out_size']) for l in layers_info)} ({output_size} classes)")

# ============================================================================
# Verify: compare manual inference vs TFLite runtime
# ============================================================================
print("\n--- Verification ---")
np.random.seed(42)
test_input = np.random.randn(1, input_size).astype(np.float32)

# TFLite inference
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])[0]

# Manual inference with extracted weights
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

x = test_input[0]
for layer in model_data['layers']:
    w = np.array(layer['weights'])
    b = np.array(layer['bias'])
    x = w @ x + b
    if layer['activation'] == 'relu':
        x = relu(x)
    elif layer['activation'] == 'softmax':
        x = softmax(x)

print(f"TFLite output (first 5):  {tflite_output[:5]}")
print(f"Manual output (first 5):  {x[:5]}")
print(f"Max error:                {np.max(np.abs(tflite_output - x)):.6f}")
print(f"Outputs match (atol=0.05): {np.allclose(tflite_output, x, atol=0.05)}")
print(f"TFLite argmax: {np.argmax(tflite_output)}, Manual argmax: {np.argmax(x)}")
