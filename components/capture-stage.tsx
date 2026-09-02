"use client";

import { Button } from "@/components/ui/button";
import {
  CAPTURE,
  FACE_MODEL_PATH,
  FLASH_MODES,
  MEDIAPIPE_WASM_CDN,
  RESOLUTIONS,
  type FlashModeId,
  type ResolutionOption,
} from "@/lib/capture-config";
import {
  canvasSizeForViewport,
  computeOval,
  guidanceText,
  isFaceCentered,
  isFaceWithinFillRange,
  type OvalGeometry,
} from "@/lib/face-geometry";
import {
  capturePhotoFromVideo,
  downloadCaptureZip,
  startMediaRecorder,
  type CapturePair,
} from "@/lib/save-captures";
import type { Detection, FaceDetector } from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";

type Phase =
  | "pick-resolution"
  | "starting"
  | "capturing"
  | "processing"
  | "done"
  | "error";

type CaptureStageProps = {
  active: boolean;
};

export function CaptureStage({ active }: CaptureStageProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectorRef = useRef<FaceDetector | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const ovalRef = useRef<OvalGeometry | null>(null);
  const closeUpRef = useRef(false);
  const validFaceRef = useRef(false);
  const countdownRef = useRef<{ active: boolean; startedAt: number }>({
    active: false,
    startedAt: 0,
  });
  const lastDetectRef = useRef(0);
  const standardRef = useRef<CapturePair>({ video: null, photo: null });
  const closeUpBlobsRef = useRef<CapturePair>({ video: null, photo: null });
  const flashColorRef = useRef<string | null>(null);
  const resolutionRef = useRef<ResolutionOption | null>(null);
  const takeLockRef = useRef(false);

  const [phase, setPhase] = useState<Phase>("pick-resolution");
  const [instruction, setInstruction] = useState("Please Select Resolution");
  const [resolutionLabel, setResolutionLabel] = useState("");
  const [progress, setProgress] = useState(0);
  const [showProgress, setShowProgress] = useState(false);
  const [showRestart, setShowRestart] = useState(false);
  const [showFlashSelect, setShowFlashSelect] = useState(false);
  const [flashMode, setFlashMode] = useState<FlashModeId>("noflash");
  const [flashVisible, setFlashVisible] = useState(false);
  const [flashColor, setFlashColor] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const stopRecorder = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") {
      rec.onstop = null;
      rec.stop();
    }
    recorderRef.current = null;
    chunksRef.current = [];
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { width, height } = canvasSizeForViewport(
      window.innerWidth,
      window.innerHeight
    );
    canvas.width = width;
    canvas.height = height;
    ovalRef.current = computeOval(width, height, closeUpRef.current);
  }, []);

  const drawFrame = useCallback((detections: Detection[]) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const oval = ovalRef.current;
    if (!canvas || !video || !oval || video.videoWidth === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const ratio = canvas.width / video.videoWidth;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    ctx.globalCompositeOperation = "destination-in";
    ctx.beginPath();
    ctx.ellipse(
      oval.centerX,
      oval.centerY,
      oval.width / 2,
      oval.height / 2,
      0,
      0,
      Math.PI * 2
    );
    ctx.fillStyle = "white";
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";

    validFaceRef.current = false;

    if (detections.length > 0) {
      const detection = [...detections].sort(
        (a, b) => (b.categories[0]?.score ?? 0) - (a.categories[0]?.score ?? 0)
      )[0];
      const confidence = detection.categories[0]?.score ?? 0;
      const box = detection.boundingBox;
      if (confidence >= CAPTURE.confidenceThreshold && box) {
        let x = box.originX * ratio;
        const y = box.originY * ratio;
        const width = box.width * ratio;
        const height = box.height * ratio;
        x = canvas.width - (x + width);
        const faceCenterX = x + width / 2;
        const faceCenterY = y + height / 2;
        const fillPercent = ((width * height) / oval.area) * 100;
        const within = isFaceWithinFillRange(width, height, oval);
        const centered = isFaceCentered(faceCenterX, faceCenterY, oval);
        validFaceRef.current = within && centered;

        if (!countdownRef.current.active) {
          setInstruction(
            validFaceRef.current
              ? "Hold still to start recording"
              : guidanceText({
                  fillPercent,
                  faceCenterX,
                  faceCenterY,
                  oval,
                  closeUp: closeUpRef.current,
                })
          );
        }
      } else if (!countdownRef.current.active) {
        setInstruction(
          closeUpRef.current
            ? "No face detected for second capture"
            : "No face detected"
        );
      }
    } else if (!countdownRef.current.active) {
      setInstruction(
        closeUpRef.current
          ? "No face detected for second capture"
          : "No face detected"
      );
    }

    ctx.beginPath();
    ctx.ellipse(
      oval.centerX,
      oval.centerY,
      oval.width / 2,
      oval.height / 2,
      0,
      0,
      Math.PI * 2
    );
    ctx.strokeStyle = validFaceRef.current ? "#4CAF50" : "#ff0000";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.globalCompositeOperation = "destination-over";
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = "source-over";
  }, []);

  const beginRecording = useCallback(() => {
    const stream = streamRef.current;
    if (!stream) return;
    stopRecorder();
    chunksRef.current = [];
    try {
      const recorder = startMediaRecorder(stream);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.start();
      recorderRef.current = recorder;
    } catch {
      setInstruction("Recording not supported in this browser");
    }
  }, [stopRecorder]);

  const completeCurrentTake = useCallback(async () => {
    if (takeLockRef.current) return;
    takeLockRef.current = true;
    countdownRef.current.active = false;
    setShowProgress(false);
    setFlashVisible(false);
    setPhase("processing");

    const recorder = recorderRef.current;
    await new Promise<void>((resolve) => {
      if (!recorder || recorder.state === "inactive") {
        resolve();
        return;
      }
      recorder.onstop = () => resolve();
      recorder.stop();
    });

    const videoEl = videoRef.current;
    if (!videoEl) {
      setInstruction("Error processing captures");
      setPhase("error");
      setShowRestart(true);
      takeLockRef.current = false;
      return;
    }

    const mimeType = recorder?.mimeType || "video/webm";
    const videoBlob = new Blob(chunksRef.current, { type: mimeType });
    recorderRef.current = null;
    chunksRef.current = [];

    try {
      const photo = await capturePhotoFromVideo(videoEl);

      if (!closeUpRef.current) {
        standardRef.current = { video: videoBlob, photo };
        closeUpRef.current = true;
        resizeCanvas();
        validFaceRef.current = false;
        countdownRef.current = { active: false, startedAt: 0 };
        setInstruction("Position for second capture (larger frame)");
        setShowProgress(false);
        setProgress(0);
        takeLockRef.current = false;
        setPhase("capturing");
        return;
      }

      closeUpBlobsRef.current = { video: videoBlob, photo };
      setInstruction("Processing second captures...");
      await downloadCaptureZip({
        standard: standardRef.current,
        closeUp: closeUpBlobsRef.current,
      });
      setInstruction("All captures saved!");
      setPhase("done");
      setShowRestart(true);
    } catch (err) {
      console.error(err);
      setInstruction("Error saving files");
      setPhase("error");
      setShowRestart(true);
    } finally {
      takeLockRef.current = false;
    }
  }, [resizeCanvas]);

  useEffect(() => {
    if (!active || phase !== "capturing") return;

    let raf = 0;
    let cancelled = false;

    const tick = () => {
      if (cancelled) return;
      const detector = detectorRef.current;
      const video = videoRef.current;
      if (!detector || !video || video.readyState < 2) {
        raf = requestAnimationFrame(tick);
        return;
      }

      const now = performance.now();
      if (now - lastDetectRef.current >= 1000 / CAPTURE.targetFps) {
        lastDetectRef.current = now;
        try {
          const result = detector.detectForVideo(video, now);
          drawFrame(result.detections ?? []);

          if (validFaceRef.current) {
            if (!countdownRef.current.active) {
              countdownRef.current = { active: true, startedAt: now };
              setShowProgress(true);
              setProgress(0);
              beginRecording();
            } else {
              const elapsed = (now - countdownRef.current.startedAt) / 1000;
              const remaining = Math.max(
                0,
                Math.ceil(CAPTURE.countdownDuration - elapsed)
              );
              const percent = Math.min(
                (elapsed / CAPTURE.countdownDuration) * 100,
                100
              );
              setProgress(percent);
              setInstruction(
                `Hold still for ${remaining} second${remaining === 1 ? "" : "s"}`
              );

              if (flashColorRef.current) {
                if (elapsed > CAPTURE.countdownDuration / 4) {
                  const blink =
                    (elapsed % CAPTURE.flashSecond) * 2 < CAPTURE.flashSecond;
                  setFlashVisible(blink);
                } else {
                  setFlashVisible(true);
                }
              }

              if (elapsed >= CAPTURE.countdownDuration) {
                void completeCurrentTake();
                return;
              }
            }
          } else if (countdownRef.current.active) {
            countdownRef.current.active = false;
            setShowProgress(false);
            setProgress(0);
            setFlashVisible(false);
            stopRecorder();
            setInstruction(
              closeUpRef.current
                ? "Position for second capture"
                : "Please keep your face in position"
            );
          }
        } catch (err) {
          console.error(err);
        }
      }

      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [active, beginRecording, completeCurrentTake, drawFrame, phase, stopRecorder]);

  const loadDetector = useCallback(async () => {
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_CDN);
    try {
      detectorRef.current = await vision.FaceDetector.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: FACE_MODEL_PATH,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
      });
    } catch {
      detectorRef.current = await vision.FaceDetector.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: FACE_MODEL_PATH,
        },
        runningMode: "VIDEO",
      });
    }
  }, []);

  const startCamera = useCallback(async (resolution: ResolutionOption) => {
    const video = videoRef.current;
    if (!video) throw new Error("Video element missing");
    streamRef.current?.getTracks().forEach((track) => track.stop());

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: resolution.width },
        height: { ideal: resolution.height },
      },
      audio: false,
    });
    streamRef.current = stream;
    video.srcObject = stream;
    await video.play();
    await new Promise<void>((resolve) => {
      if (video.readyState >= 2) resolve();
      else video.onloadedmetadata = () => resolve();
    });
    setResolutionLabel(`video: ${video.videoWidth} × ${video.videoHeight}`);
  }, []);

  const startSession = useCallback(
    async (resolution: ResolutionOption) => {
      setPhase("starting");
      setError(null);
      setInstruction("Starting camera…");
      setShowRestart(false);
      setShowProgress(false);
      setProgress(0);
      closeUpRef.current = false;
      standardRef.current = { video: null, photo: null };
      closeUpBlobsRef.current = { video: null, photo: null };
      countdownRef.current = { active: false, startedAt: 0 };
      validFaceRef.current = false;
      takeLockRef.current = false;

      try {
        await startCamera(resolution);
        resizeCanvas();
        if (!detectorRef.current) {
          setInstruction("Loading face detector…");
          await loadDetector();
        }
        if ("wakeLock" in navigator) {
          await navigator.wakeLock.request("screen").catch(() => undefined);
        }
        setPhase("capturing");
        setShowFlashSelect(true);
        setInstruction("Position your face in the oval");
      } catch (err) {
        console.error(err);
        const message =
          "Camera access error. Please reload and allow camera permissions.";
        setError(message);
        setInstruction(message);
        setPhase("error");
        setShowRestart(true);
        stopCamera();
      }
    },
    [loadDetector, resizeCanvas, startCamera, stopCamera]
  );

  const resetSession = useCallback(() => {
    stopRecorder();
    closeUpRef.current = false;
    countdownRef.current = { active: false, startedAt: 0 };
    validFaceRef.current = false;
    takeLockRef.current = false;
    setShowRestart(false);
    setShowProgress(false);
    setProgress(0);
    setFlashVisible(false);
    setError(null);
    if (!resolutionRef.current || !streamRef.current) {
      setPhase("pick-resolution");
      setShowFlashSelect(false);
      setInstruction("Please Select Resolution");
      return;
    }
    resizeCanvas();
    setInstruction("Position your face in the oval");
    setPhase("capturing");
  }, [resizeCanvas, stopRecorder]);

  useEffect(() => {
    if (!active) return;
    const onResize = () => resizeCanvas();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [active, resizeCanvas]);

  useEffect(() => {
    return () => {
      stopRecorder();
      stopCamera();
      detectorRef.current?.close();
    };
  }, [stopCamera, stopRecorder]);

  function onResolutionChange(value: string) {
    const next = RESOLUTIONS.find((item) => item.label === value);
    if (!next) return;
    resolutionRef.current = next;
    void startSession(next);
  }

  function onFlashChange(value: FlashModeId) {
    setFlashMode(value);
    const mode = FLASH_MODES.find((item) => item.id === value);
    flashColorRef.current = mode?.color ?? null;
    setFlashColor(mode?.color ?? null);
    setFlashVisible(Boolean(mode?.color));
  }

  if (!active) return null;

  return (
    <div id="appStage" className="fixed inset-0 overflow-hidden bg-black">
      <video
        ref={videoRef}
        className="hidden -scale-x-100"
        autoPlay
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-1/2 max-h-[80%] max-w-full -translate-x-1/2 aspect-[3/4]"
      />
      <div
        className="pointer-events-none absolute inset-0 z-20 transition-opacity duration-150"
        style={{
          backgroundColor: flashColor ?? "transparent",
          opacity: flashVisible && flashColor ? 0.72 : 0,
        }}
      />

      <div className="absolute inset-x-0 bottom-[max(16px,env(safe-area-inset-bottom))] z-30 flex flex-col items-center gap-2 px-4 sm:bottom-8">
        {instruction ? (
          <div className="w-[90%] max-w-md rounded-xl bg-black/80 px-4 py-3.5 text-center text-lg font-bold leading-snug text-white shadow-lg sm:text-xl">
            {instruction}
          </div>
        ) : null}

        {showRestart ? (
          <Button
            type="button"
            className="h-11 rounded-[15px] bg-[#4CAF50] px-6 text-base font-bold text-white hover:bg-[#3e8e41]"
            onClick={resetSession}
          >
            Restart
          </Button>
        ) : null}

        {showProgress ? (
          <div className="h-[15px] w-[60%] max-w-sm overflow-hidden rounded-[10px] bg-black">
            <div
              className="h-full bg-[#4CAF50] transition-[width] duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>
        ) : null}

        {resolutionLabel ? (
          <div className="w-[70%] max-w-sm rounded-xl bg-black/80 px-2 py-1 text-center text-[10px] font-bold text-white">
            {resolutionLabel}
          </div>
        ) : null}

        {phase === "pick-resolution" || phase === "starting" ? (
          <select
            aria-label="Please Select Resolution"
            className="rounded-[10px] bg-gray-500 px-3 py-2.5 text-base font-bold text-white"
            defaultValue=""
            disabled={phase === "starting"}
            onChange={(event) => onResolutionChange(event.target.value)}
          >
            <option value="" disabled>
              Please Select Resolution
            </option>
            {RESOLUTIONS.map((item) => (
              <option key={item.label} value={item.label}>
                {item.label}
              </option>
            ))}
          </select>
        ) : null}

        {showFlashSelect ? (
          <select
            aria-label="Flash mode"
            className="rounded-[10px] px-3 py-2.5 text-base font-bold text-white"
            style={{
              backgroundColor: flashColor ?? "grey",
              color: flashColor === "#ffffff" ? "#111" : "#fff",
            }}
            value={flashMode}
            onChange={(event) => onFlashChange(event.target.value as FlashModeId)}
          >
            {FLASH_MODES.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        ) : null}

        {error && phase === "error" ? (
          <p className="max-w-md text-center text-sm text-red-200">{error}</p>
        ) : null}
      </div>
    </div>
  );
}
