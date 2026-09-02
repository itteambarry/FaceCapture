"use client";

import { Button } from "@/components/ui/button";
import {
  BANKS,
  MODE_OPTIONS,
  flashCssColor,
  isBankId,
  type BankId,
  type BankTask,
} from "@/lib/bank-profiles";
import { captureSizeLabel, openUserCamera } from "@/lib/camera";
import {
  CAPTURE,
  FACE_MODEL_PATH,
  MEDIAPIPE_WASM_CDN,
  RESOLUTIONS,
  type ResolutionOption,
} from "@/lib/capture-config";
import { getFaceDirection, turnConflicts, type FaceDirection } from "@/lib/face-direction";
import {
  canvasSizeForViewport,
  computeOval,
  genericOvalParams,
  guidanceText,
  isFaceCentered,
  isFaceWithinFillRange,
  sizeTransAt,
  taskOvalParams,
  type OvalGeometry,
  type OvalParams,
} from "@/lib/face-geometry";
import {
  buildRecordingStream,
  capturePhotoFromVideo,
  downloadBankZip,
  downloadCaptureZip,
  lockManualWhiteBalance,
  startMediaRecorder,
  type BankCapture,
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

type SizePhase = "waitFrom" | "grow" | "waitTo" | "hold";

type CaptureStageProps = {
  active: boolean;
};

export function CaptureStage({ active }: CaptureStageProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<FaceDetector | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recStopRef = useRef<(() => void) | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const ovalRef = useRef<OvalGeometry | null>(null);
  const ovalParamsRef = useRef<OvalParams>(genericOvalParams(false));
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
  const genericFlashRef = useRef<string | null>(null);
  const resolutionRef = useRef<ResolutionOption | null>(null);
  const takeLockRef = useRef(false);
  const modeRef = useRef("noflash");
  const bankIdRef = useRef<BankId | null>(null);
  const taskIndexRef = useRef(0);
  const bankBlobsRef = useRef<BankCapture[]>([]);
  const currentTaskRef = useRef<BankTask | null>(null);
  const sizePhaseRef = useRef<SizePhase | null>(null);
  const sizeHoldStartRef = useRef(0);
  const sizeGrowStartRef = useRef(0);
  const fullFrameRef = useRef(false);

  const [phase, setPhase] = useState<Phase>("pick-resolution");
  const [instruction, setInstruction] = useState("Please Select Resolution");
  const [turnArrow, setTurnArrow] = useState<"left" | "right" | null>(null);
  const [resolutionLabel, setResolutionLabel] = useState("");
  const [progress, setProgress] = useState(0);
  const [showProgress, setShowProgress] = useState(false);
  const [showRestart, setShowRestart] = useState(false);
  const [showModeSelect, setShowModeSelect] = useState(false);
  const [modeId, setModeId] = useState("noflash");
  const [flashVisible, setFlashVisible] = useState(false);
  const [flashColor, setFlashColor] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [taskBadge, setTaskBadge] = useState("");
  const [faceDirection, setFaceDirection] = useState("");

  const stopRecorder = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") {
      rec.onstop = null;
      rec.stop();
    }
    recorderRef.current = null;
    chunksRef.current = [];
    recStopRef.current?.();
    recStopRef.current = null;
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
    ovalRef.current = computeOval(width, height, ovalParamsRef.current);
  }, []);

  const applyOvalParams = useCallback(
    (params: OvalParams) => {
      ovalParamsRef.current = params;
      resizeCanvas();
    },
    [resizeCanvas]
  );

  const applyTask = useCallback(
    (task: BankTask, index: number, bank: BankId) => {
      currentTaskRef.current = task;
      fullFrameRef.current = Boolean(task.showFullFramePreview);
      const trans = task.ovalSizeTrans;
      if (trans) {
        sizePhaseRef.current = "waitFrom";
        sizeHoldStartRef.current = 0;
        sizeGrowStartRef.current = 0;
        applyOvalParams(
          taskOvalParams(task, {
            widthRatio: trans.fromWidthRatio,
            heightRatio: trans.fromHeightRatio,
            minFillRatio: trans.fromMinFillRatio,
            maxFillRatio: trans.fromMaxFillRatio,
            maxOffsetX: trans.fromMaxOffsetX,
            maxOffsetY: trans.fromMaxOffsetY,
          })
        );
      } else {
        sizePhaseRef.current = null;
        applyOvalParams(taskOvalParams(task));
      }

      const css = flashCssColor(task.color);
      flashColorRef.current = css;
      setFlashColor(css);
      setFlashVisible(false);
      setTaskBadge(`${bank} · ${task.label} (${index + 1}/${BANKS[bank].length})`);
      setTurnArrow(null);
      countdownRef.current = { active: false, startedAt: 0 };
      validFaceRef.current = false;
      takeLockRef.current = false;
      setShowProgress(false);
      setProgress(0);
      setInstruction(
        task.label === "Left"
          ? "Turn your face left"
          : task.label === "Right"
            ? "Turn your face right"
            : "Position your face in the oval"
      );

      const stream = streamRef.current;
      if (stream && task.manualCameraSettings?.colorTemperature) {
        void lockManualWhiteBalance(stream, task.manualCameraSettings.colorTemperature);
      }
    },
    [applyOvalParams]
  );

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

    if (!fullFrameRef.current) {
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
    }

    validFaceRef.current = false;
    const task = currentTaskRef.current;
    const threshold = task?.confidenceThreshold ?? CAPTURE.confidenceThreshold;

    if (detections.length > 0) {
      const detection = [...detections].sort(
        (a, b) => (b.categories[0]?.score ?? 0) - (a.categories[0]?.score ?? 0)
      )[0];
      const confidence = detection.categories[0]?.score ?? 0;
      const box = detection.boundingBox;
      if (confidence >= threshold && box) {
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

        if (bankIdRef.current === "VPB") {
          const direction = getFaceDirection(detection.keypoints);
          setFaceDirection(`Your face is turning: ${direction}`);
        }

        if (!countdownRef.current.active && sizePhaseRef.current !== "grow") {
          setInstruction(
            validFaceRef.current
              ? task?.label === "Left"
                ? "Hold still to start recording"
                : task?.label === "Right"
                  ? "Hold still to start recording"
                  : "Hold still to start recording"
              : guidanceText({
                  fillPercent,
                  faceCenterX,
                  faceCenterY,
                  oval,
                  closeUp: closeUpRef.current,
                  turnLabel: task?.isTurnFace ? task.label : undefined,
                })
          );
        }
      } else if (!countdownRef.current.active) {
        setInstruction(
          task?.isTurnFace
            ? `No face detected for ${task.label} capture`
            : closeUpRef.current
              ? "No face detected for second capture"
              : "No face detected"
        );
      }
    } else if (!countdownRef.current.active) {
      setInstruction(
        task?.isTurnFace
          ? `No face detected for ${task.label} capture`
          : closeUpRef.current
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

    if (!fullFrameRef.current) {
      ctx.globalCompositeOperation = "destination-over";
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.globalCompositeOperation = "source-over";
    }
  }, []);

  const beginRecording = useCallback(() => {
    const video = videoRef.current;
    const stream = streamRef.current;
    if (!video || !stream) return;
    stopRecorder();
    chunksRef.current = [];
    try {
      if (!recCanvasRef.current) recCanvasRef.current = document.createElement("canvas");
      const built = buildRecordingStream(
        video,
        currentTaskRef.current,
        recCanvasRef.current
      );
      recStopRef.current = built.stop;
      const bitrate = currentTaskRef.current?.videoBitsPerSecond ?? 16_000_000;
      const recorder = startMediaRecorder(built.stream, bitrate);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.start();
      recorderRef.current = recorder;
    } catch {
      setInstruction("Recording not supported in this browser");
    }
  }, [stopRecorder]);

  const applyFlashForHold = useCallback((elapsed: number) => {
    const task = currentTaskRef.current;
    if (task) {
      const flashSec = task.flashSecond;
      if (flashSec === 0) setFlashVisible(Boolean(flashColorRef.current));
      else if (flashSec > 0) {
        setFlashVisible((elapsed % (flashSec * 2)) < flashSec);
      } else {
        setFlashVisible(false);
      }
      return;
    }
    if (!genericFlashRef.current) {
      setFlashVisible(false);
      return;
    }
    if (elapsed > CAPTURE.countdownDuration / 4) {
      setFlashVisible((elapsed % CAPTURE.flashSecond) * 2 < CAPTURE.flashSecond);
    } else {
      setFlashVisible(true);
    }
  }, []);

  const completeCurrentTake = useCallback(async () => {
    if (takeLockRef.current) return;
    takeLockRef.current = true;
    countdownRef.current.active = false;
    setShowProgress(false);
    setFlashVisible(false);
    setTurnArrow(null);
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
    recStopRef.current?.();
    recStopRef.current = null;

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
      const task = currentTaskRef.current;
      const photo = await capturePhotoFromVideo(
        videoEl,
        task
          ? {
              width: task.photoResolutionX ?? task.resolutionX,
              height: task.photoResolutionY ?? task.resolutionY,
            }
          : undefined
      );

      if (task && bankIdRef.current) {
        bankBlobsRef.current.push({
          config: task,
          video: task.isVideo ? videoBlob : null,
          photo,
        });
        const nextIndex = taskIndexRef.current + 1;
        const bank = bankIdRef.current;
        const tasks = BANKS[bank];
        if (nextIndex >= tasks.length) {
          setInstruction("Processing captures...");
          await downloadBankZip(bankBlobsRef.current);
          setInstruction("All captures completed");
          setPhase("done");
          setShowRestart(true);
          takeLockRef.current = false;
          return;
        }
        taskIndexRef.current = nextIndex;
        applyTask(tasks[nextIndex], nextIndex, bank);
        setPhase("capturing");
        takeLockRef.current = false;
        return;
      }

      if (!closeUpRef.current) {
        standardRef.current = { video: videoBlob, photo };
        closeUpRef.current = true;
        applyOvalParams(genericOvalParams(true));
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
  }, [applyOvalParams, applyTask]);

  const startHold = useCallback(
    (now: number) => {
      countdownRef.current = { active: true, startedAt: now };
      setShowProgress(true);
      setProgress(0);
      if (!recorderRef.current || recorderRef.current.state === "inactive") {
        beginRecording();
      }
    },
    [beginRecording]
  );

  const tickHold = useCallback(
    (now: number) => {
      const task = currentTaskRef.current;
      const duration = task?.recordSecond ?? CAPTURE.countdownDuration;
      const elapsed = (now - countdownRef.current.startedAt) / 1000;
      const remaining = Math.max(0, Math.ceil(duration - elapsed));
      setProgress(Math.min((elapsed / duration) * 100, 100));

      if (task?.label === "Left") {
        setInstruction(`Face left slowly for ${remaining} second${remaining === 1 ? "" : "s"}`);
        setTurnArrow("left");
      } else if (task?.label === "Right") {
        setInstruction(`Face right slowly for ${remaining} second${remaining === 1 ? "" : "s"}`);
        setTurnArrow("right");
      } else {
        setInstruction(`Hold still for ${remaining} second${remaining === 1 ? "" : "s"}`);
        setTurnArrow(null);
      }

      applyFlashForHold(elapsed);
      if (elapsed >= duration) void completeCurrentTake();
    },
    [applyFlashForHold, completeCurrentTake]
  );

  const cancelHold = useCallback(() => {
    countdownRef.current.active = false;
    setShowProgress(false);
    setProgress(0);
    setTurnArrow(null);
    const task = currentTaskRef.current;
    if (!task?.enableFlashCoverageGate) setFlashVisible(false);
    stopRecorder();
    if (task?.ovalSizeTrans) {
      sizePhaseRef.current = "waitFrom";
      sizeHoldStartRef.current = 0;
      applyOvalParams(
        taskOvalParams(task, {
          widthRatio: task.ovalSizeTrans.fromWidthRatio,
          heightRatio: task.ovalSizeTrans.fromHeightRatio,
          minFillRatio: task.ovalSizeTrans.fromMinFillRatio,
          maxFillRatio: task.ovalSizeTrans.fromMaxFillRatio,
          maxOffsetX: task.ovalSizeTrans.fromMaxOffsetX,
          maxOffsetY: task.ovalSizeTrans.fromMaxOffsetY,
        })
      );
      setInstruction("Position your face in the oval");
      return;
    }
    setInstruction(
      closeUpRef.current
        ? "Position for second capture"
        : task?.isTurnFace
          ? `Turn your face ${task.label.toLowerCase()}`
          : "Please keep your face in position"
    );
  }, [applyOvalParams, stopRecorder]);

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
          const detections = result.detections ?? [];
          drawFrame(detections);

          const task = currentTaskRef.current;
          const top = detections[0];
          const direction: FaceDirection = getFaceDirection(top?.keypoints);
          if (task && turnConflicts(task.label, direction)) {
            if (countdownRef.current.active) cancelHold();
            raf = requestAnimationFrame(tick);
            return;
          }

          if (task?.enableFlashCoverageGate && !countdownRef.current.active) {
            setFlashVisible(validFaceRef.current && Boolean(flashColorRef.current));
          }

          const trans = task?.ovalSizeTrans;
          const sizePhase = sizePhaseRef.current;

          if (task && trans && sizePhase && sizePhase !== "hold") {
            if (sizePhase === "grow") {
              const t = trans.time <= 0 ? 1 : (now - sizeGrowStartRef.current) / trans.time;
              applyOvalParams(sizeTransAt(task, trans, t));
              setInstruction("Move closer to the camera");
              if (t >= 1) {
                sizePhaseRef.current = "waitTo";
                sizeHoldStartRef.current = 0;
                applyOvalParams(taskOvalParams(task));
                setInstruction("Move closer — fill the larger oval");
              } else if (!top) {
                cancelHold();
              }
            } else if (sizePhase === "waitFrom") {
              if (!validFaceRef.current) {
                sizeHoldStartRef.current = 0;
                setInstruction("Position your face in the oval");
              } else {
                if (!recorderRef.current || recorderRef.current.state === "inactive") {
                  beginRecording();
                }
                if (!sizeHoldStartRef.current) sizeHoldStartRef.current = now;
                const held = now - sizeHoldStartRef.current;
                if (held < trans.fromRecordTime) {
                  const remain = Math.ceil((trans.fromRecordTime - held) / 1000);
                  setInstruction(`Hold still for ${remain} second${remain === 1 ? "" : "s"}`);
                  setShowProgress(true);
                  setProgress((held / trans.fromRecordTime) * 100);
                } else {
                  sizeHoldStartRef.current = 0;
                  sizeGrowStartRef.current = now;
                  sizePhaseRef.current = "grow";
                  setShowProgress(false);
                  setInstruction("Move closer to the camera");
                }
              }
            } else if (sizePhase === "waitTo") {
              if (!validFaceRef.current) {
                sizeHoldStartRef.current = 0;
                setInstruction("Move closer — fill the larger oval");
              } else if (trans.toRecordTime <= 0) {
                sizePhaseRef.current = "hold";
                startHold(now);
              } else {
                if (!sizeHoldStartRef.current) sizeHoldStartRef.current = now;
                const held = now - sizeHoldStartRef.current;
                if (held >= trans.toRecordTime) {
                  sizePhaseRef.current = "hold";
                  startHold(now);
                }
              }
            }
          } else if (validFaceRef.current) {
            if (!countdownRef.current.active) startHold(now);
            else tickHold(now);
          } else if (countdownRef.current.active) {
            cancelHold();
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
  }, [
    active,
    applyOvalParams,
    beginRecording,
    cancelHold,
    drawFrame,
    phase,
    startHold,
    tickHold,
  ]);

  const loadDetector = useCallback(async () => {
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_CDN);
    try {
      detectorRef.current = await vision.FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL_PATH, delegate: "GPU" },
        runningMode: "VIDEO",
      });
    } catch {
      detectorRef.current = await vision.FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL_PATH },
        runningMode: "VIDEO",
      });
    }
  }, []);

  const startCamera = useCallback(async (resolution: ResolutionOption) => {
    const video = videoRef.current;
    if (!video) throw new Error("Video element missing");
    streamRef.current?.getTracks().forEach((track) => track.stop());

    const stream = await openUserCamera(resolution);
    streamRef.current = stream;
    video.srcObject = stream;
    await video.play();
    await new Promise<void>((resolve) => {
      if (video.readyState >= 2) resolve();
      else video.onloadedmetadata = () => resolve();
    });
    await new Promise((resolve) => window.setTimeout(resolve, 80));
    const track = stream.getVideoTracks()[0];
    const settings = track?.getSettings?.() ?? {};
    const cameraWidth = settings.width ?? video.videoWidth;
    const cameraHeight = settings.height ?? video.videoHeight;
    setResolutionLabel(captureSizeLabel(resolution, cameraWidth, cameraHeight));
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
      currentTaskRef.current = null;
      bankIdRef.current = null;
      ovalParamsRef.current = genericOvalParams(false);
      fullFrameRef.current = false;

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
        setShowModeSelect(true);
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
    setTurnArrow(null);
    setError(null);
    if (!resolutionRef.current || !streamRef.current) {
      setPhase("pick-resolution");
      setShowModeSelect(false);
      setInstruction("Please Select Resolution");
      return;
    }
    const bank = bankIdRef.current;
    if (bank) {
      bankBlobsRef.current = [];
      taskIndexRef.current = 0;
      applyTask(BANKS[bank][0], 0, bank);
      setPhase("capturing");
      return;
    }
    applyOvalParams(genericOvalParams(false));
    setInstruction("Position your face in the oval");
    setPhase("capturing");
  }, [applyOvalParams, applyTask, stopRecorder]);

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

  function onModeChange(value: string) {
    setModeId(value);
    modeRef.current = value;
    stopRecorder();
    countdownRef.current = { active: false, startedAt: 0 };
    closeUpRef.current = false;
    takeLockRef.current = false;
    setShowRestart(false);
    setShowProgress(false);
    setProgress(0);
    setFaceDirection("");
    setTurnArrow(null);

    if (isBankId(value)) {
      bankIdRef.current = value;
      bankBlobsRef.current = [];
      taskIndexRef.current = 0;
      applyTask(BANKS[value][0], 0, value);
      setPhase("capturing");
      return;
    }

    bankIdRef.current = null;
    currentTaskRef.current = null;
    sizePhaseRef.current = null;
    fullFrameRef.current = false;
    setTaskBadge("");
    const option = MODE_OPTIONS.find((item) => item.id === value);
    genericFlashRef.current = option?.color ?? null;
    flashColorRef.current = option?.color ?? null;
    setFlashColor(option?.color ?? null);
    setFlashVisible(Boolean(option?.color));
    applyOvalParams(genericOvalParams(false));
    setInstruction("Position your face in the oval");
    setPhase("capturing");
  }

  const selectColor = flashColor ?? "grey";

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

      {faceDirection ? (
        <div className="absolute top-3 left-3 z-40 rounded-lg bg-black/70 px-2 py-1 text-sm font-bold text-white">
          {faceDirection}
        </div>
      ) : null}

      {taskBadge ? (
        <div className="absolute top-3 right-3 z-40 rounded-lg bg-black/70 px-2 py-1 text-xs font-bold tracking-wide text-emerald-200">
          {taskBadge}
        </div>
      ) : null}

      <div className="absolute inset-x-0 bottom-[max(16px,env(safe-area-inset-bottom))] z-30 flex flex-col items-center gap-2 px-4 sm:bottom-8">
        {instruction ? (
          <div className="w-[90%] max-w-md rounded-xl bg-black/80 px-4 py-3.5 text-center text-lg font-bold leading-snug text-white shadow-lg sm:text-xl">
            <div>{instruction}</div>
            {turnArrow === "left" ? (
              <span className="mt-1 block animate-[slide_0.5s_infinite] text-4xl text-[#4CAF50]">
                {"<<<"}
              </span>
            ) : null}
            {turnArrow === "right" ? (
              <span className="mt-1 block animate-[slide_0.5s_infinite] text-4xl text-[#4CAF50]">
                {">>>"}
              </span>
            ) : null}
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

        {showModeSelect ? (
          <select
            aria-label="Bank or flash mode"
            className="rounded-[10px] px-3 py-2.5 text-base font-bold text-white"
            style={{
              backgroundColor: selectColor,
              color: selectColor === "#ffffff" || selectColor === "#FFCF0C" ? "#111" : "#fff",
            }}
            value={modeId}
            onChange={(event) => onModeChange(event.target.value)}
          >
            {MODE_OPTIONS.map((item) => (
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
