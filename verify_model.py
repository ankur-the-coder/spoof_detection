"""
Verify the ONNX model's class order and preprocessing.
This uses the EXACT same preprocessing as the reference repo (suriAI/face-antispoof-onnx).
Run this to determine the correct class order for YOUR model file.

Usage:
    python verify_model.py
    python verify_model.py --image path/to/face.jpg
"""

import cv2
import numpy as np
import onnxruntime as ort
import sys
import argparse


def preprocess_reference(img: np.ndarray, model_img_size: int = 128) -> np.ndarray:
    """Exact preprocessing from reference repo (src/inference/preprocess.py)"""
    new_size = model_img_size
    old_size = img.shape[:2]  # (height, width)

    ratio = float(new_size) / max(old_size)
    scaled_shape = tuple([int(x * ratio) for x in old_size])

    # Use INTER_AREA for downscaling (most common), LANCZOS4 for upscaling
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (scaled_shape[1], scaled_shape[0]), interpolation=interpolation)

    # Letterbox padding with reflect
    delta_w = new_size - scaled_shape[1]
    delta_h = new_size - scaled_shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    # CHW format, float32, [0, 1]
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    return img


def crop_reference(img: np.ndarray, bbox: tuple, expansion: float = 1.5) -> np.ndarray:
    """Exact crop from reference repo (src/inference/preprocess.py)"""
    original_height, original_width = img.shape[:2]
    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        raise ValueError("Invalid bbox dimensions")

    max_dim = max(w, h)
    center_x = x + w / 2
    center_y = y + h / 2

    x = int(center_x - max_dim * expansion / 2)
    y = int(center_y - max_dim * expansion / 2)
    crop_size = int(max_dim * expansion)

    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(original_width, x + crop_size)
    crop_y2 = min(original_height, y + crop_size)

    top_pad = int(max(0, -y))
    left_pad = int(max(0, -x))
    bottom_pad = int(max(0, (y + crop_size) - original_height))
    right_pad = int(max(0, (x + crop_size) - original_width))

    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    else:
        img = np.zeros((0, 0, 3), dtype=img.dtype)

    result = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REFLECT_101)
    return result


def detect_face_simple(img):
    """Simple face detection using OpenCV's Haar cascade."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    return faces


def run_inference(model_path, img_rgb, bbox):
    """Run inference with reference preprocessing and print detailed results."""
    x, y, w, h = bbox
    
    # Crop with reference method
    face_crop = crop_reference(img_rgb, (x, y, w, h), expansion=1.5)
    
    # Show the crop
    cv2.imshow("Face Crop (RGB)", cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
    
    # Preprocess with reference method
    preprocessed = preprocess_reference(face_crop, 128)
    batch_input = np.expand_dims(preprocessed, axis=0)  # Add batch dim
    
    # Run model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    logits = session.run([output_name], {input_name: batch_input})[0][0]
    
    # Print raw logits
    print(f"\n{'='*60}")
    print(f"Face at ({x}, {y}, {w}, {h})")
    print(f"Raw logits: [{logits[0]:.6f}, {logits[1]:.6f}]")
    print(f"")
    
    # Try BOTH class orders
    # Order A: index 0 = spoof, index 1 = real (Silent-Face-Anti-Spoofing original)
    exp_a = np.exp(logits - np.max(logits))
    prob_a = exp_a / exp_a.sum()
    print(f"Interpretation A (0=spoof, 1=real):")
    print(f"  Prob SPOOF = {prob_a[0]*100:.2f}%, Prob REAL = {prob_a[1]*100:.2f}%")
    print(f"  Result: {'REAL' if prob_a[1] > 0.5 else 'SPOOF'} ({max(prob_a)*100:.2f}%)")
    
    # Order B: index 0 = real, index 1 = spoof (suriAI reference)
    print(f"\nInterpretation B (0=real, 1=spoof):")
    print(f"  Prob REAL = {prob_a[0]*100:.2f}%, Prob SPOOF = {prob_a[1]*100:.2f}%")
    print(f"  Result: {'REAL' if prob_a[0] > 0.5 else 'SPOOF'} ({max(prob_a)*100:.2f}%)")
    
    # logit diff method (reference repo style, assuming 0=real 1=spoof)
    logit_diff = logits[0] - logits[1]
    print(f"\nLogit diff (logits[0] - logits[1]): {logit_diff:.6f}")
    print(f"  If 0=real,1=spoof: {'REAL' if logit_diff >= 0 else 'SPOOF'}")
    print(f"  If 0=spoof,1=real: {'REAL' if logit_diff < 0 else 'SPOOF'}")
    print(f"{'='*60}")
    
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--model", type=str, default="public/onnx-assets/best_model_quantized.onnx")
    args = parser.parse_args()
    
    print(f"Model: {args.model}")
    
    # Load model info
    session = ort.InferenceSession(args.model)
    print(f"Input: {session.get_inputs()[0].name}, shape={session.get_inputs()[0].shape}")
    print(f"Output: {session.get_outputs()[0].name}, shape={session.get_outputs()[0].shape}")
    
    if args.image:
        # Process single image
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not load {args.image}")
            sys.exit(1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces = detect_face_simple(frame_rgb)
        if len(faces) == 0:
            print("No faces detected!")
            sys.exit(1)
        
        for bbox in faces:
            run_inference(args.model, frame_rgb, tuple(bbox))
        
        cv2.waitKey(0)
    else:
        # Webcam mode
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)
        
        print("\nPress SPACE to capture and analyze, Q to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow("Webcam - Press SPACE to capture", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detect_face_simple(frame_rgb)
                
                if len(faces) == 0:
                    print("No faces detected! Try again.")
                else:
                    print(f"\nDetected {len(faces)} face(s)")
                    for i, bbox in enumerate(faces):
                        print(f"\n--- Face {i+1} ---")
                        run_inference(args.model, frame_rgb, tuple(bbox))
                    cv2.waitKey(500)
            
            elif key == ord('q'):
                break
        
        cap.release()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
