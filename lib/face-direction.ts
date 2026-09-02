export type FaceDirection = "Left" | "Right" | "Up" | "Down" | "frontal" | "unknown";

type Point = { x: number; y: number };

export function getFaceDirection(keypoints: Point[] | undefined): FaceDirection {
  if (!keypoints || keypoints.length < 4) return "unknown";
  const [rightEye, leftEye, noseTip, mouth] = keypoints;
  if (!rightEye || !leftEye || !noseTip || !mouth) return "unknown";

  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const horizontalOffset = noseTip.x - eyeMidX;
  const HORIZONTAL_THR = 0.02;
  if (horizontalOffset > HORIZONTAL_THR) return "Left";
  if (horizontalOffset < -HORIZONTAL_THR) return "Right";

  const eyeMidY = (leftEye.y + rightEye.y) / 2;
  const noseToEyes = Math.abs(noseTip.y - eyeMidY);
  const noseToMouth = Math.abs(noseTip.y - mouth.y);
  const verticalRatio = noseToEyes === 0 ? 1 : noseToMouth / noseToEyes;
  if (verticalRatio < 0.8) return "Down";
  if (verticalRatio > 1.5) return "Up";
  return "frontal";
}

export function turnConflicts(taskLabel: string, direction: FaceDirection) {
  return (
    (taskLabel === "Left" && direction === "Right") ||
    (taskLabel === "Right" && direction === "Left")
  );
}
