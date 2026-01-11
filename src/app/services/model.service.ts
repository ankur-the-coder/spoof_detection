import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-cpu';
import * as blazeface from '@tensorflow-models/blazeface';

@Injectable({
    providedIn: 'root'
})
export class ModelService {
    private spoofSession: ort.InferenceSession | null = null;
    private faceDetector: blazeface.BlazeFaceModel | null = null;
    private isModelsLoaded = false;

    async loadModels() {
        if (this.isModelsLoaded) return;

        try {
            console.log('Initializing ONNX Runtime...');
            ort.env.wasm.wasmPaths = '/onnx-assets/';

            console.log('Loading Anti-Spoofing Model...');
            const modelResponse = await fetch('/onnx-assets/antispoof.onnx');
            if (!modelResponse.ok) {
                throw new Error(`Failed to fetch model: ${modelResponse.status} ${modelResponse.statusText}`);
            }
            const modelBuffer = await modelResponse.arrayBuffer();

            // Set up ONNX Runtime threads
            // Note: SharedArrayBuffer (threads) requires Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers.
            // If not present, this falls back to single thread regardless of setting.
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

            console.log('Initializing TensorFlow.js (WASM backend)...');
            // Force WASM backend for stability and CPU performance
            // Needs: npm install @tensorflow/tfjs-backend-wasm
            await tf.setBackend('wasm');
            await tf.ready();

            this.faceDetector = await blazeface.load();

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

    getFaceDetector(): blazeface.BlazeFaceModel | null {
        return this.faceDetector;
    }

    isLoaded(): boolean {
        return this.isModelsLoaded;
    }
}
