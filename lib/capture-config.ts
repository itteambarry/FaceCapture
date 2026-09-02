import { withBase } from "@/lib/base-path";

export const APP_NAME = "BB Face Capture";
export const APP_SHORT_NAME = "FaceCapture";

export const RESOLUTIONS = [
  { label: "1920 x 1440", width: 1920, height: 1440 },
  { label: "1200 x 1600", width: 1200, height: 1600 },
  { label: "1080 x 1440", width: 1080, height: 1440 },
  { label: "1280 x 720", width: 1280, height: 720 },
] as const;

export type ResolutionOption = (typeof RESOLUTIONS)[number];

export const FLASH_MODES = [
  { id: "noflash", label: "noflash", color: null },
  { id: "red", label: "red", color: "#ff0000" },
  { id: "orange", label: "orange", color: "#ff8000" },
  { id: "white", label: "white", color: "#ffffff" },
  { id: "blue", label: "blue", color: "#0000ff" },
  { id: "green", label: "green", color: "#00ff00" },
] as const;

export type FlashModeId = (typeof FLASH_MODES)[number]["id"];

export const CAPTURE = {
  confidenceThreshold: 0.8,
  maxOffsetX: 25,
  maxOffsetY: 30,
  minFillRatio: 0.85,
  maxFillRatio: 1.1,
  ovalWidthRatio: 0.5,
  ovalHeightRatio: 0.6,
  widerOvalWidthRatio: 0.65,
  widerOvalHeightRatio: 0.75,
  ovalCenterYRatio: 2.0,
  ovalOffsetYRatio: 0.125,
  countdownDuration: 5,
  flashSecond: 0.8,
  targetFps: 30,
} as const;

export const MEDIAPIPE_WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
export const FACE_MODEL_PATH = withBase("/wasm/blaze_face_short_range.tflite");
