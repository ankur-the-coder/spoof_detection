import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';
import { ModelService } from './model.service';

@Injectable({
    providedIn: 'root'
})
export class AntiSpoofService {
    constructor(private modelService: ModelService) { }

    // Reusable canvases (avoid GC pressure)
    private stage1Canvas = document.createElement('canvas');
    private finalCanvas = document.createElement('canvas');

    // Pre-allocated buffer for preprocessing (128x128x3 = 49152 floats)
    private preprocessBuffer = new Float32Array(3 * 128 * 128);

    // Debug photo capture
    private capturedPhotos: string[] = [];

    getCapturedPhotos(): string[] {
        return this.capturedPhotos;
    }

    async predict(video: HTMLVideoElement, detection: any): Promise<{ score: number; isReal: boolean }> {
        const session = this.modelService.getSpoofSession();
        if (!session) throw new Error('Anti-spoofing session not initialized');

        // Expects detection object with normalized boundingBox
        const bbox = detection.boundingBox;
        if (!bbox) throw new Error('No bounding box in detection');

        const x = bbox.originX * video.videoWidth;
        const y = bbox.originY * video.videoHeight;
        const width = bbox.width * video.videoWidth;
        const height = bbox.height * video.videoHeight;

        // 1. Expand face bounding box (Reference repo uses 1.5)
        const expansionFactor = 1.5;
        const newWidth = width * expansionFactor;
        const newHeight = height * expansionFactor;

        let newX = x - (newWidth - width) / 2;
        let newY = y - (newHeight - height) / 2;

        // Ensure square crop (use larger dimension) to match model expectation
        const maxSide = Math.max(newWidth, newHeight);
        newX = newX - (maxSide - newWidth) / 2;
        newY = newY - (maxSide - newHeight) / 2;

        // 2. Crop face from video (Stage 1: Accurate Crop at Full Resolution)
        // Reuse canvas - just resize if needed
        if (this.stage1Canvas.width !== maxSide || this.stage1Canvas.height !== maxSide) {
            this.stage1Canvas.width = maxSide;
            this.stage1Canvas.height = maxSide;
        }
        const stage1Ctx = this.stage1Canvas.getContext('2d', { alpha: false });
        if (!stage1Ctx) throw new Error('Could not get stage1 canvas context');

        // Fill with black to handle potential out-of-bounds areas (padding)
        stage1Ctx.fillStyle = 'black';
        stage1Ctx.fillRect(0, 0, maxSide, maxSide);

        stage1Ctx.imageSmoothingEnabled = true;
        stage1Ctx.imageSmoothingQuality = 'high';

        stage1Ctx.drawImage(
            video,
            newX, newY, maxSide, maxSide, // Source (x,y,w,h)
            0, 0, maxSide, maxSide        // Destination
        );

        // 3. High-Quality Downscale using native Lanczos (createImageBitmap)
        // resizeQuality: 'high' uses Lanczos interpolation in modern browsers
        const bitmap = await createImageBitmap(this.stage1Canvas, {
            resizeWidth: 128,
            resizeHeight: 128,
            resizeQuality: 'high'
        });

        // Reuse final canvas
        if (this.finalCanvas.width !== 128 || this.finalCanvas.height !== 128) {
            this.finalCanvas.width = 128;
            this.finalCanvas.height = 128;
        }
        const ctx = this.finalCanvas.getContext('2d', { alpha: false });
        if (!ctx) throw new Error('Could not get final canvas context');

        ctx.drawImage(bitmap, 0, 0);
        bitmap.close(); // Release memory

        // Store debug photo (Limit to 50)
        if (this.capturedPhotos.length < 50) {
            this.capturedPhotos.push(this.finalCanvas.toDataURL('image/jpeg', 0.95));
        }

        // 4. Get image data and preprocess
        const imageData = ctx.getImageData(0, 0, 128, 128).data;
        const input = this.preprocess(imageData);

        // 5. Run inference
        const feeds: any = {};
        feeds[session.inputNames[0]] = input;
        const output = await session.run(feeds);
        const logits = output[session.outputNames[0]].data as Float32Array;

        // 6. Post-process (real - spoof)
        // logits format: [spoof, real] (Fixed class order)
        const score = logits[1] - logits[0];
        const isReal = score > 0;

        return { score, isReal };
    }

    private preprocess(data: Uint8ClampedArray): ort.Tensor {
        // Reuse pre-allocated buffer
        const float32Data = this.preprocessBuffer;

        // ImageNet normalization constants
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];

        const pixelCount = 128 * 128;
        for (let i = 0; i < pixelCount; i++) {
            const r = data[i * 4] / 255.0;
            const g = data[i * 4 + 1] / 255.0;
            const b = data[i * 4 + 2] / 255.0;

            // CHW format: R, G, B (Reference uses RGB input)
            float32Data[i] = (r - mean[0]) / std[0];
            float32Data[i + pixelCount] = (g - mean[1]) / std[1];
            float32Data[i + 2 * pixelCount] = (b - mean[2]) / std[2];
        }

        return new ort.Tensor('float32', float32Data, [1, 3, 128, 128]);
    }
}
