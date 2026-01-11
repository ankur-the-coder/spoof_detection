import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, OnDestroy, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CameraService } from './services/camera.service';
import { ModelService } from './services/model.service';
import { AntiSpoofService } from './services/antispoof.service';


@Component({
    selector: 'app-root',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, AfterViewInit, OnDestroy {
    @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
    @ViewChild('overlayCanvas') overlayCanvas!: ElementRef<HTMLCanvasElement>;

    isLoaded = false;
    fps = 0;
    cpuInfo = 'Detecting...';
    gpuInfo = 'No GPU detected';
    provider = 'WASM';

    private lastTime = 0;
    private animationId: number | null = null;
    showDebug = true;

    @HostListener('document:keydown.t', ['$event'])
    toggleDebug(event: KeyboardEvent) {
        this.showDebug = !this.showDebug;
    }

    constructor(
        private cameraService: CameraService,
        private modelService: ModelService,
        private antiSpoofService: AntiSpoofService
    ) { }

    async ngOnInit() {
        this.detectHardware();
    }

    async ngAfterViewInit() {
        try {
            await this.modelService.loadModels();
            this.isLoaded = true;

            await this.cameraService.startCamera(this.videoElement.nativeElement);
            this.startDetectionLoop();
        } catch (error) {
            console.error('Initialization failed:', error);
        }
    }

    ngOnDestroy() {
        if (this.animationId) cancelAnimationFrame(this.animationId);
        this.cameraService.stopCamera();
    }

    private detectHardware() {
        const cores = navigator.hardwareConcurrency || 4;
        this.cpuInfo = `CPUs (Logical Cores): ${cores}`;

        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
            const debugInfo = (gl as any).getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
                this.gpuInfo = (gl as any).getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            }
        } else {
            this.gpuInfo = 'No WebGL Support';
        }
    }

    private startDetectionLoop() {
        const loop = async () => {
            const video = this.videoElement.nativeElement;
            const faceDetector = this.modelService.getFaceDetector();

            if (video.readyState >= 2 && faceDetector) {
                const startTime = performance.now();

                try {
                    // BlazeFace estimateFaces returns pixel coordinates
                    // Pass existing video element directly
                    const predictions = await faceDetector.estimateFaces(video, false);
                    await this.handleDetectionResults(predictions);

                    const endTime = performance.now();
                    const delta = endTime - this.lastTime;
                    if (delta > 0) {
                        this.fps = 1000 / delta;
                    }
                    this.lastTime = endTime;
                } catch (err) {
                    console.error('Face detection error:', err);
                }
            }

            this.animationId = requestAnimationFrame(loop);
        };

        this.animationId = requestAnimationFrame(loop);
    }



    // ... (existing methods: constructor, ngOnInit, ngAfterViewInit, ngOnDestroy, detectHardware, startDetectionLoop)

    private async handleDetectionResults(predictions: any[]) {
        const video = this.videoElement.nativeElement;
        const canvas = this.overlayCanvas.nativeElement;
        const ctx = canvas.getContext('2d');

        if (!ctx) return;

        // Match canvas size to video
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // If debug is disabled, don't draw anything (or just draw clean video? User said "toggle debug", implies overlay info)
        // Usually toggle debug toggles the TEXT info. The bounding boxes are main functionality.
        // But the screenshot says "Press 't' to toggle debug" effectively checking the overlay.
        // I will hide the bounding boxes too if debug is off? Or just the overlay?
        // Let's assume toggle debug toggles the ENTIRE overlay (boxes + stats) or just stats?
        // "there is a text on screen which says 'press t to toggle debug' but nothing happen"
        // I will make it toggle the overlay.

        for (const pred of predictions) {
            try {
                // BlazeFace returns [x, y] arrays for topLeft and bottomRight
                const start = pred.topLeft as [number, number];
                const end = pred.bottomRight as [number, number];
                const probability = (pred.probability as number[])[0];

                // Raw detected dimensions
                const rawWidth = end[0] - start[0];
                const rawHeight = end[1] - start[1];
                const rawX = start[0];
                const rawY = start[1];

                // Make it SQUARE
                const side = Math.max(rawWidth, rawHeight);
                const centerX = rawX + rawWidth / 2;
                const centerY = rawY + rawHeight / 2;

                const squareX = centerX - side / 2;
                const squareY = centerY - side / 2;

                // 1. Pass LOGICAL (unflipped) coordinates to AntiSpoofService
                // Normalize for service
                const detection = {
                    boundingBox: {
                        originX: squareX / video.videoWidth,
                        originY: squareY / video.videoHeight,
                        width: side / video.videoWidth,
                        height: side / video.videoHeight
                    },
                    categories: [{ score: probability }]
                };

                const result = await this.antiSpoofService.predict(video, detection);

                // 2. Calculate MIRRORED coordinates for Drawing (because video is flipped via CSS)
                // mirroredX = width - x - width
                const mirroredX = video.videoWidth - squareX - side;

                // 3. Scale down visual box (0.8x) as per user request, while keeping inference box same
                const visualScale = 0.8;
                const visualSide = side * visualScale;
                const visualX = mirroredX + (side - visualSide) / 2;
                const visualY = squareY + (side - visualSide) / 2;

                this.drawResult(ctx, { x: visualX, y: visualY, width: visualSide, height: visualSide }, result);
            } catch (err) {
                console.error('Anti-spoofing error:', err);
            }
        }
    }

    private drawResult(ctx: CanvasRenderingContext2D, bbox: { x: number, y: number, width: number, height: number }, result: { score: number, isReal: boolean }) {
        if (!this.showDebug) return;

        const color = result.isReal ? '#00ff00' : '#ff0000';
        const labelText = `${result.isReal ? 'REAL' : 'SPOOF'}: ${result.score.toFixed(2)}`;

        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

        ctx.fillStyle = color;
        ctx.font = 'bold 28px Courier New';
        ctx.fillText(labelText, bbox.x, bbox.y - 10);
    }
}
