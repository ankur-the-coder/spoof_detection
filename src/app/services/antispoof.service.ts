import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';
import { ModelService } from './model.service';

@Injectable({
    providedIn: 'root'
})
export class AntiSpoofService {
    constructor(private modelService: ModelService) { }

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

        // 2. Crop face from video
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Could not get canvas context');

        // Draw the expanded face crop to the 128x128 canvas
        ctx.drawImage(
            video,
            newX, newY, maxSide, maxSide, // Source (x,y,w,h) - can be outside bounds (Canvas handles clipping)
            0, 0, 128, 128 // Destination
        );

        // 3. Preprocess image data
        const imageData = ctx.getImageData(0, 0, 128, 128).data;
        const input = this.preprocess(imageData, 128, 128);

        // 4. Run inference
        const feeds: any = {};
        feeds[session.inputNames[0]] = input;
        const output = await session.run(feeds);
        const logits = output[session.outputNames[0]].data as Float32Array;

        // 5. Post-process (real - spoof)
        // logits format: [real, spoof]
        const score = logits[0] - logits[1];
        const isReal = score > 0;

        return { score, isReal };
    }

    private preprocess(data: Uint8ClampedArray, width: number, height: number): ort.Tensor {
        const float32Data = new Float32Array(3 * width * height);

        // Normalize and convert RGB to BGR, then to CHW format
        for (let i = 0; i < width * height; i++) {
            const r = data[i * 4] / 255.0;
            const g = data[i * 4 + 1] / 255.0;
            const b = data[i * 4 + 2] / 255.0;

            // CHW format: B, G, R
            float32Data[i] = b; // Blue channel
            float32Data[i + width * height] = g; // Green channel
            float32Data[i + 2 * width * height] = r; // Red channel
        }

        return new ort.Tensor('float32', float32Data, [1, 3, height, width]);
    }
}
