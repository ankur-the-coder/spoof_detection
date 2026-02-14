"""
Non-interactive model class order verification.
Tests with synthetic inputs and a quick webcam capture.
"""
import numpy as np
import onnxruntime as ort
import cv2

MODEL_PATH = "public/onnx-assets/best_model_quantized.onnx"

def preprocess_exact(img_rgb, model_img_size=128):
    """Exact reference repo preprocessing on an RGB image."""
    old_size = img_rgb.shape[:2]
    ratio = float(model_img_size) / max(old_size)
    scaled_shape = tuple([int(x * ratio) for x in old_size])
    interp = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img_rgb, (scaled_shape[1], scaled_shape[0]), interpolation=interp)
    
    delta_w = model_img_size - scaled_shape[1]
    delta_h = model_img_size - scaled_shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0

def run_model(session, input_tensor):
    input_name = session.get_inputs()[0].name
    batch = np.expand_dims(input_tensor, axis=0)
    logits = session.run(None, {input_name: batch})[0][0]
    return logits

print(f"Loading model: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH)
print(f"Input: {session.get_inputs()[0].name}, shape={session.get_inputs()[0].shape}")
print(f"Output: {session.get_outputs()[0].name}, shape={session.get_outputs()[0].shape}")

# Test 1: Random noise (should be spoof - not a real face)
print("\n--- Test 1: Random noise ---")
noise = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
tensor = noise.transpose(2, 0, 1).astype(np.float32) / 255.0
logits = run_model(session, tensor)
print(f"logits = [{logits[0]:.6f}, {logits[1]:.6f}]")
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / exp_logits.sum()
print(f"softmax = [{probs[0]:.6f}, {probs[1]:.6f}]")

# Test 2: Black image
print("\n--- Test 2: Black image ---")
black = np.zeros((128, 128, 3), dtype=np.uint8)
tensor = black.transpose(2, 0, 1).astype(np.float32) / 255.0
logits = run_model(session, tensor)
print(f"logits = [{logits[0]:.6f}, {logits[1]:.6f}]")

# Test 3: White image
print("\n--- Test 3: White image ---")
white = np.full((128, 128, 3), 255, dtype=np.uint8)
tensor = white.transpose(2, 0, 1).astype(np.float32) / 255.0
logits = run_model(session, tensor)
print(f"logits = [{logits[0]:.6f}, {logits[1]:.6f}]")

# Test 4: Webcam capture
print("\n--- Test 4: Webcam capture ---")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Skip first few frames (camera warmup)
    for _ in range(10):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            print(f"Face detected at ({x}, {y}, {w}, {h})")
            
            # Crop with expansion
            max_dim = max(w, h)
            cx, cy = x + w//2, y + h//2
            expansion = 1.5
            crop_size = int(max_dim * expansion)
            x1 = max(0, cx - crop_size // 2)
            y1 = max(0, cy - crop_size // 2)
            x2 = min(frame_rgb.shape[1], x1 + crop_size)
            y2 = min(frame_rgb.shape[0], y1 + crop_size)
            
            face_crop = frame_rgb[y1:y2, x1:x2]
            
            # Save the crop for inspection
            cv2.imwrite("debug_face_crop.png", cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            print(f"Saved face crop to debug_face_crop.png ({face_crop.shape})")
            
            # Preprocess using reference method
            tensor_rgb = preprocess_exact(face_crop, 128)
            logits_rgb = run_model(session, tensor_rgb)
            exp_logits = np.exp(logits_rgb - np.max(logits_rgb))
            probs_rgb = exp_logits / exp_logits.sum()
            
            print(f"\nWith RGB input (reference repo style):")
            print(f"  logits = [{logits_rgb[0]:.6f}, {logits_rgb[1]:.6f}]")
            print(f"  softmax = [{probs_rgb[0]:.6f}, {probs_rgb[1]:.6f}]")
            print(f"  If 0=spoof,1=real => {'REAL' if probs_rgb[1] > 0.5 else 'SPOOF'} ({probs_rgb[1]*100:.1f}% real)")
            print(f"  If 0=real,1=spoof => {'REAL' if probs_rgb[0] > 0.5 else 'SPOOF'} ({probs_rgb[0]*100:.1f}% real)")
            
            # Also test with BGR input to compare
            face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            tensor_bgr = preprocess_exact(face_crop_bgr, 128)
            logits_bgr = run_model(session, tensor_bgr)
            exp_logits_bgr = np.exp(logits_bgr - np.max(logits_bgr))
            probs_bgr = exp_logits_bgr / exp_logits_bgr.sum()
            
            print(f"\nWith BGR input (original Silent-Face style):")
            print(f"  logits = [{logits_bgr[0]:.6f}, {logits_bgr[1]:.6f}]")
            print(f"  softmax = [{probs_bgr[0]:.6f}, {probs_bgr[1]:.6f}]")
            print(f"  If 0=spoof,1=real => {'REAL' if probs_bgr[1] > 0.5 else 'SPOOF'} ({probs_bgr[1]*100:.1f}% real)")
            print(f"  If 0=real,1=spoof => {'REAL' if probs_bgr[0] > 0.5 else 'SPOOF'} ({probs_bgr[0]*100:.1f}% real)")
        else:
            print("No face detected in webcam frame")
    else:
        print("Failed to capture frame")
else:
    print("Could not open camera")

print("\n\nDONE")
