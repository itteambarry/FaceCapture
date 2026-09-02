import JSZip from "jszip";

export type CapturePair = {
  video: Blob | null;
  photo: Blob | null;
};

export async function downloadCaptureZip(opts: {
  standard: CapturePair;
  closeUp: CapturePair;
}) {
  const timestamp = new Date().toISOString().replace(/:/g, "-");
  const zip = new JSZip();

  const add = (blob: Blob | null, name: string) => {
    if (blob) zip.file(name, blob);
  };

  const videoExt = (blob: Blob | null) =>
    blob && blob.type.includes("mp4") ? "mp4" : "webm";

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

  const zipBlob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(zipBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `face-captures-${timestamp}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

export function capturePhotoFromVideo(video: HTMLVideoElement): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
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

export function startMediaRecorder(stream: MediaStream): MediaRecorder {
  const mp4 = "video/mp4;codecs=avc1.42E01E, mp4a.40.2";
  try {
    return new MediaRecorder(stream, {
      mimeType: mp4,
      videoBitsPerSecond: 8_000_000,
    });
  } catch {
    return new MediaRecorder(stream, { mimeType: "video/webm" });
  }
}
