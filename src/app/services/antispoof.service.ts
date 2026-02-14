import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';
import { ModelService } from './model.service';

export interface SpoofResult {
    score: number;
    isReal: boolean;
    label: string;
    realLogit: number;
    spoofLogit: number;
}

@Injectable({
    providedIn: 'root'
})
export class AntiSpoofService {
    constructor(private modelService: ModelService) { }

    // Reusable canvases (avoid GC pressure)
    private cropCanvas = document.createElement('canvas');
    private resizeCanvas = document.createElement('canvas');
    private midCanvas = document.createElement('canvas'); // Intermediate canvas for downscaling

    // Pre-allocated buffer for preprocessing (128x128x3 = 49152 floats)
    private preprocessBuffer = new Float32Array(3 * 128 * 128);

    // Inference lock to prevent concurrent ONNX session.run() calls
    private inferencing = false;

    /**
     * Predict liveness for a single face.
     * 
     * @param video - The video element to crop from
     * @param bbox - Pixel-coordinate bounding box { x, y, width, height }
     * @returns SpoofResult with score, isReal, and label
     */
    async predict(video: HTMLVideoElement, bbox: { x: number; y: number; width: number; height: number }): Promise<SpoofResult> {
        const session = this.modelService.getSpoofSession();
        if (!session) throw new Error('Anti-spoofing session not initialized');

        // Wait if another inference is running (ONNX WASM session is not reentrant)
        while (this.inferencing) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        this.inferencing = true;

        try {
            // 1. Calculate the Square Crop Region (with 1.5x expansion)
            //    Reference: "max_dim * expansion"
            const side = Math.max(bbox.width, bbox.height);
            const centerX = bbox.x + bbox.width / 2;
            const centerY = bbox.y + bbox.height / 2;
            const expansionFactor = 1.5;
            const cropSize = Math.round(side * expansionFactor);

            // 2. Calculate crop coordinates (can be out of video bounds)
            const x1 = Math.round(centerX - cropSize / 2);
            const y1 = Math.round(centerY - cropSize / 2);
            // We want to capture 'cropSize' width/height from the video source.
            // But we must clip to what is actually available in the video frame.

            const visibleX1 = Math.max(0, x1);
            const visibleY1 = Math.max(0, y1);
            const visibleX2 = Math.min(video.videoWidth, x1 + cropSize);
            const visibleY2 = Math.min(video.videoHeight, y1 + cropSize);

            const visibleW = visibleX2 - visibleX1;
            const visibleH = visibleY2 - visibleY1;

            if (visibleW <= 0 || visibleH <= 0) {
                return { score: 0, isReal: false, label: 'SPOOF', realLogit: 0, spoofLogit: 0 };
            }

            // 3. Extract the visible Face Region into 'cropCanvas'
            //    We resize 'cropCanvas' to match the visible region size exactly.
            if (this.cropCanvas.width !== visibleW || this.cropCanvas.height !== visibleH) {
                this.cropCanvas.width = visibleW;
                this.cropCanvas.height = visibleH;
            }
            const cropCtx = this.cropCanvas.getContext('2d', { alpha: false });
            if (!cropCtx) throw new Error('Could not get crop canvas context');

            // disable smoothing for the initial crop extraction to preserve raw pixels? 
            // Actually standard is fine here as it is 1:1 copy usually.
            cropCtx.drawImage(video, visibleX1, visibleY1, visibleW, visibleH, 0, 0, visibleW, visibleH);

            // 4. Prepare 'resizeCanvas' (128x128) - The Model Input
            const targetSize = 128;
            if (this.resizeCanvas.width !== targetSize || this.resizeCanvas.height !== targetSize) {
                this.resizeCanvas.width = targetSize;
                this.resizeCanvas.height = targetSize;
            }
            const resizeCtx = this.resizeCanvas.getContext('2d', { alpha: false });
            if (!resizeCtx) throw new Error('Could not get resize canvas context');

            // Fill with Black (Letterboxing background)
            resizeCtx.fillStyle = 'black';
            resizeCtx.fillRect(0, 0, targetSize, targetSize);
            resizeCtx.imageSmoothingEnabled = true;
            resizeCtx.imageSmoothingQuality = 'high';

            // 5. Letterbox Logic: Fit the 'cropCanvas' contents into 'resizeCanvas'
            //    We must simulate where the 'visible region' sits within the 'ideal (expanded) crop'
            //    and then map that to the 128x128 space.

            // Scaling factor from "Ideal Crop Size" -> "Target Size"
            const scale = targetSize / cropSize;

            // Calculate where the visible portion should be drawn on the target canvas
            // Offset of valid data relative to the ideal crop start (x1, y1)
            const offsetX = (visibleX1 - x1) * scale;
            const offsetY = (visibleY1 - y1) * scale;

            // Scaled dimensions of the visible data
            const drawW = visibleW * scale;
            const drawH = visibleH * scale;

            // 6. Draw with high-quality downscaling
            //    If the source is significantly larger than the target destination, use stepping.
            //    Threshold: if we are shrinking by more than 2x.
            if (visibleW > drawW * 2) {
                // Two-step resize
                // Step A: Resize to 2x the destination size (or slightly larger)
                const midW = Math.round(drawW * 2);
                const midH = Math.round(drawH * 2);

                if (this.midCanvas.width !== midW || this.midCanvas.height !== midH) {
                    this.midCanvas.width = midW;
                    this.midCanvas.height = midH;
                }
                const midCtx = this.midCanvas.getContext('2d', { alpha: false });
                if (!midCtx) throw new Error('Could not get mid canvas context');

                midCtx.imageSmoothingEnabled = true;
                midCtx.imageSmoothingQuality = 'high';
                midCtx.drawImage(this.cropCanvas, 0, 0, visibleW, visibleH, 0, 0, midW, midH);

                // Step B: Resize from Mid to Target
                resizeCtx.drawImage(this.midCanvas, 0, 0, midW, midH, offsetX, offsetY, drawW, drawH);
            } else {
                // Direct resize
                resizeCtx.drawImage(this.cropCanvas, 0, 0, visibleW, visibleH, offsetX, offsetY, drawW, drawH);
            }

            // 7. Preprocess and Inference
            const imageData = resizeCtx.getImageData(0, 0, targetSize, targetSize).data;
            const input = this.preprocess(imageData);

            // Run ONNX inference
            const feeds: Record<string, ort.Tensor> = {};
            feeds[session.inputNames[0]] = input;
            const output = await session.run(feeds);
            const logits = output[session.outputNames[0]].data as Float32Array;

            // Post-process: logits[0] = REAL, logits[1] = SPOOF
            const realLogit = logits[0];
            const spoofLogit = logits[1];

            // Softmax
            const maxLogit = Math.max(realLogit, spoofLogit);
            const expReal = Math.exp(realLogit - maxLogit);
            const expSpoof = Math.exp(spoofLogit - maxLogit);
            const sumExp = expReal + expSpoof;

            const probReal = expReal / sumExp;
            const probSpoof = expSpoof / sumExp;

            const isReal = probReal > probSpoof;
            const label = isReal ? 'REAL' : 'SPOOF';
            // Use probability as score
            const score = isReal ? probReal : probSpoof;

            return { score, isReal, label, realLogit, spoofLogit };
        } finally {
            this.inferencing = false;
        }
    }

    /**
     * Convert RGBA pixel data to RGB CHW float32 tensor normalized to [0, 1].
     * Channel order: R, G, B (model trained on RGB input).
     * Reference: img.transpose(2, 0, 1).astype(np.float32) / 255.0
     */
    private preprocess(data: Uint8ClampedArray): ort.Tensor {
        const float32Data = this.preprocessBuffer;
        const pixelCount = 128 * 128;

        // Pass 1: Find per-channel min/max for adaptive contrast stretching
        // This normalizes dim indoor images to full range, fixing misclassification
        let rMin = 255, rMax = 0, gMin = 255, gMax = 0, bMin = 255, bMax = 0;
        for (let i = 0; i < pixelCount; i++) {
            const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
            if (r < rMin) rMin = r; if (r > rMax) rMax = r;
            if (g < gMin) gMin = g; if (g > gMax) gMax = g;
            if (b < bMin) bMin = b; if (b > bMax) bMax = b;
        }

        // Compute ranges with safety guard (avoid division by near-zero on degenerate inputs)
        const MIN_RANGE = 10;
        const rRange = (rMax - rMin) >= MIN_RANGE ? (rMax - rMin) : 255;
        const gRange = (gMax - gMin) >= MIN_RANGE ? (gMax - gMin) : 255;
        const bRange = (bMax - bMin) >= MIN_RANGE ? (bMax - bMin) : 255;
        const rOff = rRange === 255 ? 0 : rMin;
        const gOff = gRange === 255 ? 0 : gMin;
        const bOff = bRange === 255 ? 0 : bMin;

        // Pass 2: Normalize with per-channel contrast stretching (CHW, RGB order)
        for (let i = 0; i < pixelCount; i++) {
            const r = (data[i * 4] - rOff) / rRange;
            const g = (data[i * 4 + 1] - gOff) / gRange;
            const b = (data[i * 4 + 2] - bOff) / bRange;

            float32Data[i] = r;                // Channel 0: Red
            float32Data[i + pixelCount] = g;   // Channel 1: Green
            float32Data[i + 2 * pixelCount] = b; // Channel 2: Blue
        }

        return new ort.Tensor('float32', float32Data, [1, 3, 128, 128]);
    }
}
