import { CAPTURE } from "@/lib/capture-config";

export type OvalGeometry = {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  offsetCenterY: number;
  area: number;
  minFaceArea: number;
  maxFaceArea: number;
};

export function canvasSizeForViewport(innerWidth: number, innerHeight: number) {
  const aspectRatio = 3 / 4;
  const padding = 20;
  const maxWidth = innerWidth - padding * 2;
  const maxHeight = innerHeight - padding * 2;

  let width: number;
  let height: number;

  if (maxWidth / aspectRatio <= maxHeight) {
    width = maxWidth;
    height = width / aspectRatio;
  } else {
    height = maxHeight;
    width = height * aspectRatio;
  }

  return {
    width: Math.floor(width),
    height: Math.floor(height),
  };
}

export function computeOval(
  canvasWidth: number,
  canvasHeight: number,
  closeUp: boolean
): OvalGeometry {
  const centerX = canvasWidth / 2;
  const centerY = canvasHeight / CAPTURE.ovalCenterYRatio;
  const width = canvasWidth * (closeUp ? CAPTURE.widerOvalWidthRatio : CAPTURE.ovalWidthRatio);
  const height = canvasHeight * (closeUp ? CAPTURE.widerOvalHeightRatio : CAPTURE.ovalHeightRatio);
  const offsetCenterY = height * CAPTURE.ovalOffsetYRatio;
  const area = Math.PI * (width / 2) * (height / 2);

  return {
    centerX,
    centerY,
    width,
    height,
    offsetCenterY,
    area,
    minFaceArea: CAPTURE.minFillRatio * area,
    maxFaceArea: CAPTURE.maxFillRatio * area,
  };
}

export function isFaceCentered(
  faceCenterX: number,
  faceCenterY: number,
  oval: OvalGeometry
) {
  const distanceX = Math.abs(faceCenterX - oval.centerX);
  const distanceY = Math.abs(faceCenterY - (oval.centerY + oval.offsetCenterY));
  return distanceX <= CAPTURE.maxOffsetX && distanceY <= CAPTURE.maxOffsetY;
}

export function isFaceWithinFillRange(faceWidth: number, faceHeight: number, oval: OvalGeometry) {
  const faceArea = faceWidth * faceHeight;
  return faceArea >= oval.minFaceArea && faceArea <= oval.maxFaceArea;
}

export function guidanceText(opts: {
  fillPercent: number;
  faceCenterX: number;
  faceCenterY: number;
  oval: OvalGeometry;
  closeUp: boolean;
}): string {
  const { fillPercent, faceCenterX, faceCenterY, oval, closeUp } = opts;
  let instruction = "";

  if (fillPercent < CAPTURE.minFillRatio * 100) {
    instruction = "Move closer to the camera";
  } else if (fillPercent > CAPTURE.maxFillRatio * 100) {
    instruction = "Move further from the camera";
  } else {
    if (faceCenterX < oval.centerX - CAPTURE.maxOffsetX) {
      instruction = "Move right";
    } else if (faceCenterX > oval.centerX + CAPTURE.maxOffsetX) {
      instruction = "Move left";
    }
    if (faceCenterY < oval.centerY + oval.offsetCenterY - CAPTURE.maxOffsetY) {
      instruction += instruction ? " & down" : "Move down";
    } else if (faceCenterY > oval.centerY + oval.offsetCenterY + CAPTURE.maxOffsetY) {
      instruction += instruction ? " & up" : "Move up";
    }
  }

  if (instruction) return instruction;
  return closeUp
    ? "Position your face in the wider oval"
    : "Position your face in the oval";
}
