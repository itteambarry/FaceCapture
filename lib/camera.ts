import type { ResolutionOption } from "@/lib/capture-config";

export async function openUserCamera(resolution: ResolutionOption): Promise<MediaStream> {
  const isPortrait = window.matchMedia("(orientation: portrait)").matches;
  const nativeAspect = resolution.width / resolution.height;
  const aspectRatio = isPortrait ? nativeAspect : 1 / nativeAspect;

  const attempts: MediaStreamConstraints[] = [
    {
      audio: false,
      video: {
        width: { ideal: resolution.width },
        height: { ideal: resolution.height },
        aspectRatio: { exact: aspectRatio },
        frameRate: { min: 24, ideal: 30, max: 30 },
        facingMode: "user",
      },
    },
    {
      audio: false,
      video: {
        width: { ideal: resolution.width },
        height: { ideal: resolution.height },
        aspectRatio: { exact: aspectRatio },
        frameRate: { max: 30 },
        facingMode: "user",
      },
    },
    {
      audio: false,
      video: {
        width: { ideal: resolution.width },
        height: { ideal: resolution.height },
        facingMode: "user",
      },
    },
    {
      audio: false,
      video: { facingMode: "user" },
    },
  ];

  let stream: MediaStream | null = null;
  let lastError: unknown = null;

  for (const constraints of attempts) {
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      break;
    } catch (error) {
      lastError = error;
    }
  }

  if (!stream) {
    throw lastError instanceof Error ? lastError : new Error("Camera access error");
  }

  return stream;
}

export function captureSizeLabel(
  selected: ResolutionOption,
  cameraWidth: number,
  cameraHeight: number
) {
  if (cameraWidth === selected.width && cameraHeight === selected.height) {
    return `video: ${selected.width} × ${selected.height}`;
  }
  return `video: ${selected.width} × ${selected.height}  (camera ${cameraWidth} × ${cameraHeight})`;
}
