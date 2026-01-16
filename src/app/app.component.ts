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
    private antiSpoofRunning = false; // Throttle anti-spoof to prevent queue buildup

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
                try {
                    const startTimeMs = performance.now();

                    // MediaPipe detectForVideo
                    const result = faceDetector.detectForVideo(video, startTimeMs);

                    if (result.detections.length > 0) {
                        await this.handleDetectionResults(result.detections);
                    } else {
                        // Clear canvas if no faces
                        const canvas = this.overlayCanvas.nativeElement;
                        const ctx = canvas.getContext('2d');
                        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
                    }

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

    private async handleDetectionResults(detections: any[]) {
        const video = this.videoElement.nativeElement;
        const canvas = this.overlayCanvas.nativeElement;
        const ctx = canvas.getContext('2d');

        if (!ctx) return;

        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Process all faces concurrently
        // We map each detection to a promise that resolves to its result + config
        const promises = detections.map(async (det) => {
            try {
                // MediaPipe returns pixel coordinates in 'boundingBox'
                const { originX, originY, width, height } = det.boundingBox;

                // Make it SQUARE for Anti-Spoof Service (maintain original logic)
                const centerX = originX + width / 2;
                const centerY = originY + height / 2;
                const side = Math.max(width, height);

                const squareX = centerX - side / 2;
                const squareY = centerY - side / 2;

                // Prepare normalized detection for Service
                const detectionForService = {
                    boundingBox: {
                        originX: squareX / video.videoWidth,
                        originY: squareY / video.videoHeight,
                        width: side / video.videoWidth,
                        height: side / video.videoHeight
                    }
                    // categories not strictly needed by service if we trust bbox, but service might access it?
                    // Checking service: it only uses boundingBox.
                };

                // Run Anti-Spoof Prediction
                const result = await this.antiSpoofService.predict(video, detectionForService);

                return {
                    originalBox: { x: originX, y: originY, width, height },
                    squareBox: { x: squareX, y: squareY, width: side, height: side },
                    result
                };
            } catch (err) {
                console.error('Anti-spoofing error for face:', err);
                return null;
            }
        });

        const results = await Promise.all(promises);

        // Draw all results
        results.forEach(item => {
            if (item) {
                // Calculate MIRRORED coordinates for Drawing (because video is flipped via CSS)
                // Note: The bounding box 'x' from MediaPipe is relative to the video source (not flipped).
                // CSS flip means we need to draw at (Width - x - w).

                const { x, y, width, height } = item.squareBox;

                const mirroredX = video.videoWidth - x - width;

                // Scale down visual box (0.8x) as per user request
                const visualScale = 0.8;
                const visualSide = width * visualScale;
                const visualX = mirroredX + (width - visualSide) / 2;
                const visualY = y + (height - visualSide) / 2;

                this.drawResult(ctx, { x: visualX, y: visualY, width: visualSide, height: visualSide }, item.result);
            }
        });
    }

    isModalOpen = false;
    capturedPhotos: string[] = [];

    openPhotoModal() {
        console.log('Opening photo modal...');
        this.capturedPhotos = this.antiSpoofService.getCapturedPhotos();
        console.log('Captured photos count:', this.capturedPhotos.length);
        this.isModalOpen = true;
    }

    closePhotoModal() {
        this.isModalOpen = false;
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
