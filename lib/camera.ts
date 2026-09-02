import type { ResolutionOption } from "@/lib/capture-config";

export async function openUserCamera(resolution: ResolutionOption): Promise<MediaStream> {
  const isPortrait = window.matchMedia("(orientation: portrait)").matches;
  const nativeAspect = resolution.width / resolution.height;
  const aspectRatio = isPortrait ? nativeAspect : 1 / nativeAspect;

  const attempts: MediaStreamConstraints[] = [
    {
      audio: false,
      video: {
        width: { exact: resolution.width },
        height: { exact: resolution.height },
        frameRate: { ideal: 30, max: 30 },
        facingMode: { ideal: "user" },
      },
    },
    {
      audio: false,
      video: {
        width: { min: resolution.width, ideal: resolution.width },
        height: { min: resolution.height, ideal: resolution.height },
        facingMode: { ideal: "user" },
      },
    },
    {
      audio: false,
      video: {
        width: { ideal: resolution.width },
        height: { ideal: resolution.height },
        aspectRatio: { exact: aspectRatio },
        frameRate: { min: 24, ideal: 30, max: 30 },
        facingMode: { ideal: "user" },
      },
    },
    {
      audio: false,
      video: {
        width: { ideal: resolution.width },
        height: { ideal: resolution.height },
        facingMode: { ideal: "user" },
      },
    },
    {
      audio: false,
      video: { facingMode: { ideal: "user" } },
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

  const track = stream.getVideoTracks()[0];
  if (track?.applyConstraints) {
    try {
      await track.applyConstraints({
        width: { exact: resolution.width },
        height: { exact: resolution.height },
      });
    } catch {
      try {
        await track.applyConstraints({
          width: { ideal: resolution.width },
          height: { ideal: resolution.height },
        });
      } catch {
        /* keep whatever the device granted */
      }
    }
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
