import type { BankTask } from "@/lib/bank-profiles";
import JSZip from "jszip";

export type CapturePair = {
  video: Blob | null;
  photo: Blob | null;
};

export type BankCapture = {
  config: BankTask;
  video: Blob | null;
  photo: Blob | null;
};

function videoExt(blob: Blob | null) {
  return blob && blob.type.includes("mp4") ? "mp4" : "webm";
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

export async function downloadCaptureZip(opts: {
  standard: CapturePair;
  closeUp: CapturePair;
}) {
  const timestamp = new Date().toISOString().replace(/:/g, "-");
  const zip = new JSZip();
  const add = (blob: Blob | null, name: string) => {
    if (blob) zip.file(name, blob);
  };

  add(
    opts.standard.video,
    `face-capture-standard-${timestamp}.${videoExt(opts.standard.video)}`
  );
  add(opts.standard.photo, `face-capture-standard-${timestamp}.jpg`);
  add(
    opts.closeUp.video,
    `face-capture-wide-${timestamp}.${videoExt(opts.closeUp.video)}`
  );
  add(opts.closeUp.photo, `face-capture-wide-${timestamp}.jpg`);

  triggerDownload(await zip.generateAsync({ type: "blob" }), `face-captures-${timestamp}.zip`);
}

export async function downloadBankZip(captures: BankCapture[]) {
  const timestamp = new Date().toISOString().replace(/:/g, "-");
  const zip = new JSZip();

  captures.forEach((item, index) => {
    const { config, video, photo } = item;
    if (video && config.isVideo) {
      zip.file(
        `${config.zipFilename}-${timestamp}-${index}-${config.label}.${videoExt(video)}`,
        video
      );
    }
    if (photo) {
      zip.file(`${config.zipFilename}-${timestamp}-${index}-${config.label}.jpg`, photo);
    }
  });

  triggerDownload(await zip.generateAsync({ type: "blob" }), `face-captures-${timestamp}.zip`);
}

export function capturePhotoFromVideo(
  video: HTMLVideoElement,
  size?: { width?: number; height?: number }
): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = size?.width || video.videoWidth;
  canvas.height = size?.height || video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.reject(new Error("Canvas unavailable"));
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Photo capture failed"));
      },
      "image/jpeg",
      0.95
    );
  });
}

export function startMediaRecorder(
  stream: MediaStream,
  bitsPerSecond = 8_000_000
): MediaRecorder {
  const mp4 = "video/mp4;codecs=avc1.42E01E, mp4a.40.2";
  try {
    return new MediaRecorder(stream, {
      mimeType: mp4,
      videoBitsPerSecond: bitsPerSecond,
    });
  } catch {
    return new MediaRecorder(stream, {
      mimeType: "video/webm",
      videoBitsPerSecond: bitsPerSecond,
    });
  }
}

export function buildRecordingStream(
  video: HTMLVideoElement,
  task: BankTask | null,
  recCanvas: HTMLCanvasElement
): { stream: MediaStream; stop: () => void } {
  const rotate = Boolean(task?.videoRotate90CW);
  const targetW = task?.videoResolutionX;
  const targetH = task?.videoResolutionY;

  if (!rotate && !targetW && !targetH) {
    return { stream: video.srcObject as MediaStream, stop: () => undefined };
  }

  const outW = targetW || (rotate ? video.videoHeight : video.videoWidth);
  const outH = targetH || (rotate ? video.videoWidth : video.videoHeight);
  recCanvas.width = outW;
  recCanvas.height = outH;
  const ctx = recCanvas.getContext("2d");
  if (!ctx) {
    return { stream: video.srcObject as MediaStream, stop: () => undefined };
  }

  let raf = 0;
  let lastPaint = 0;
  const draw = (timestamp: number) => {
    if (timestamp - lastPaint < 30) {
      raf = requestAnimationFrame(draw);
      return;
    }
    lastPaint = timestamp;
    ctx.save();
    if (rotate) {
      ctx.translate(recCanvas.width, 0);
      ctx.rotate(Math.PI / 2);
      ctx.drawImage(video, 0, 0, outH, outW);
    } else {
      ctx.drawImage(video, 0, 0, outW, outH);
    }
    ctx.restore();
    raf = requestAnimationFrame(draw);
  };
  raf = requestAnimationFrame(draw);

  return {
    stream: recCanvas.captureStream(),
    stop: () => cancelAnimationFrame(raf),
  };
}

export async function lockManualWhiteBalance(
  stream: MediaStream,
  colorTemperature: number
) {
  const track = stream.getVideoTracks()[0];
  if (!track?.getCapabilities || !track.applyConstraints) return;
  const caps = track.getCapabilities() as MediaTrackCapabilities & {
    whiteBalanceMode?: string[];
    colorTemperature?: { min: number; max: number };
  };
  if (!caps.whiteBalanceMode?.includes("manual") || !caps.colorTemperature) return;
  const t = Math.max(
    caps.colorTemperature.min,
    Math.min(caps.colorTemperature.max, colorTemperature)
  );
  try {
    await track.applyConstraints({
      advanced: [{ whiteBalanceMode: "manual", colorTemperature: t } as MediaTrackConstraintSet],
    });
  } catch {
    /* device may reject manual WB */
  }
}
