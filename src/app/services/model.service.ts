import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

@Injectable({
    providedIn: 'root'
})
export class ModelService {
    private spoofSession: ort.InferenceSession | null = null;
    private faceDetector: FaceDetector | null = null;
    private isModelsLoaded = false;

    async loadModels() {
        if (this.isModelsLoaded) return;

        try {
            console.log('Initializing MediaPipe Face Detector...');
            const vision = await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
            );

            this.faceDetector = await FaceDetector.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: '/models/blaze_face_short_range.tflite',
                    delegate: 'GPU'
                },
                runningMode: 'VIDEO',
                minDetectionConfidence: 0.5,
                minSuppressionThreshold: 0.3
            });

            console.log('Initializing ONNX Runtime...');
            ort.env.wasm.wasmPaths = '/onnx-assets/';

            console.log('Loading Anti-Spoofing Model...');
            const modelResponse = await fetch('/onnx-assets/antispoof.onnx');
            if (!modelResponse.ok) {
                throw new Error(`Failed to fetch model: ${modelResponse.status} ${modelResponse.statusText}`);
            }
            const modelBuffer = await modelResponse.arrayBuffer();

            ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
            ort.env.wasm.simd = true;
            ort.env.logLevel = 'error';

            this.spoofSession = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ['wasm', 'cpu'],
                graphOptimizationLevel: 'all',
                executionMode: 'parallel',
                interOpNumThreads: navigator.hardwareConcurrency || 4,
                intraOpNumThreads: navigator.hardwareConcurrency || 4
            });

            this.isModelsLoaded = true;
            console.log('Models loaded successfully');
        } catch (error) {
            console.error('Error loading models:', error);
            throw error;
        }
    }

    getSpoofSession(): ort.InferenceSession | null {
        return this.spoofSession;
    }

    getFaceDetector(): FaceDetector | null {
        return this.faceDetector;
    }

    isLoaded(): boolean {
        return this.isModelsLoaded;
    }
}
