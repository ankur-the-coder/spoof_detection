import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, OnDestroy, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CameraService } from './services/camera.service';
import { ModelService } from './services/model.service';
import { AntiSpoofService, SpoofResult } from './services/antispoof.service';

// Temporal smoothing: track predictions per face position
interface TrackedFace {
    x: number;
    y: number;
    width: number;
    height: number;
    history: SpoofResult[];
    lastSeen: number;
}

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

    // Temporal smoothing: track faces across frames
    private trackedFaces: TrackedFace[] = [];
    private readonly SMOOTHING_WINDOW = 5;   // Average over N frames
    private readonly FACE_MATCH_THRESHOLD = 100; // pixels distance to match tracked face
    private readonly MIN_FACE_SIZE = 40;      // Minimum face dimension in pixels

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

                    // MediaPipe face detection
                    const result = faceDetector.detectForVideo(video, startTimeMs);

                    if (result.detections.length > 0) {
                        await this.handleDetectionResults(result.detections);
                    } else {
                        // Clear canvas if no faces detected
                        const canvas = this.overlayCanvas.nativeElement;
                        const ctx = canvas.getContext('2d');
                        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
                        // Clear tracked faces when no faces are detected
                        this.trackedFaces = [];
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

        const now = performance.now();

        // Process faces SEQUENTIALLY to avoid ONNX session race conditions
        const currentFrameFaces: Array<{
            box: { x: number; y: number; width: number; height: number };
            result: SpoofResult;
        }> = [];

        for (const det of detections) {
            try {
                const { originX, originY, width, height } = det.boundingBox;

                // Filter out tiny faces (likely false positives)
                if (width < this.MIN_FACE_SIZE || height < this.MIN_FACE_SIZE) {
                    continue;
                }

                // Run anti-spoof prediction
                const result = await this.antiSpoofService.predict(video, {
                    x: originX,
                    y: originY,
                    width,
                    height
                });

                currentFrameFaces.push({
                    box: { x: originX, y: originY, width, height },
                    result
                });
            } catch (err) {
                console.error('Anti-spoofing error for face:', err);
            }
        }

        // Update tracked faces with temporal smoothing
        const usedTracked = new Set<number>();

        for (const face of currentFrameFaces) {
            const centerX = face.box.x + face.box.width / 2;
            const centerY = face.box.y + face.box.height / 2;

            // Find matching tracked face
            let bestMatch = -1;
            let bestDist = Infinity;
            for (let i = 0; i < this.trackedFaces.length; i++) {
                if (usedTracked.has(i)) continue;
                const tracked = this.trackedFaces[i];
                const trackedCX = tracked.x + tracked.width / 2;
                const trackedCY = tracked.y + tracked.height / 2;
                const dist = Math.hypot(centerX - trackedCX, centerY - trackedCY);
                if (dist < this.FACE_MATCH_THRESHOLD && dist < bestDist) {
                    bestDist = dist;
                    bestMatch = i;
                }
            }

            if (bestMatch >= 0) {
                // Update existing tracked face
                const tracked = this.trackedFaces[bestMatch];
                tracked.x = face.box.x;
                tracked.y = face.box.y;
                tracked.width = face.box.width;
                tracked.height = face.box.height;
                tracked.history.push(face.result);
                if (tracked.history.length > this.SMOOTHING_WINDOW) {
                    tracked.history.shift();
                }
                tracked.lastSeen = now;
                usedTracked.add(bestMatch);
            } else {
                // New face
                this.trackedFaces.push({
                    x: face.box.x,
                    y: face.box.y,
                    width: face.box.width,
                    height: face.box.height,
                    history: [face.result],
                    lastSeen: now
                });
            }
        }

        // Remove stale tracked faces (not seen for 500ms)
        this.trackedFaces = this.trackedFaces.filter(f => now - f.lastSeen < 500);

        // Draw results with smoothed predictions
        for (const tracked of this.trackedFaces) {
            if (tracked.history.length === 0) continue;

            // Compute smoothed result: majority vote + averaged confidence
            const realCount = tracked.history.filter(h => h.isReal).length;
            const totalCount = tracked.history.length;
            const smoothedIsReal = realCount > totalCount / 2;

            // Average scores for the smoothed class
            const relevantScores = tracked.history
                .filter(h => h.isReal === smoothedIsReal)
                .map(h => h.score);
            const avgScore = relevantScores.length > 0
                ? relevantScores.reduce((a, b) => a + b, 0) / relevantScores.length
                : tracked.history[tracked.history.length - 1].score;

            const smoothedResult: SpoofResult = {
                isReal: smoothedIsReal,
                score: avgScore,
                label: smoothedIsReal ? 'REAL' : 'SPOOF',
                realLogit: tracked.history[tracked.history.length - 1].realLogit,
                spoofLogit: tracked.history[tracked.history.length - 1].spoofLogit
            };

            // Mirror X coordinate (video is CSS-flipped with scaleX(-1))
            const mirroredX = video.videoWidth - tracked.x - tracked.width;

            this.drawResult(ctx, {
                x: mirroredX,
                y: tracked.y,
                width: tracked.width,
                height: tracked.height
            }, smoothedResult);
        }
    }

    private drawResult(
        ctx: CanvasRenderingContext2D,
        bbox: { x: number; y: number; width: number; height: number },
        result: SpoofResult
    ) {
        if (!this.showDebug) return;

        const color = result.isReal ? '#00ff00' : '#ff0000';
        const labelText = `${result.label}: ${(result.score * 100).toFixed(1)}%`;

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

        // Background for label text
        ctx.font = 'bold 20px Courier New';
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(bbox.x, bbox.y - 26, textWidth + 8, 24);

        ctx.fillStyle = color;
        ctx.fillText(labelText, bbox.x + 4, bbox.y - 8);
    }
}
