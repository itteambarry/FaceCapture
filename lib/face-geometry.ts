import { CAPTURE } from "@/lib/capture-config";
import type { BankTask, OvalSizeTrans } from "@/lib/bank-profiles";

export type OvalGeometry = {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  offsetCenterY: number;
  area: number;
  minFaceArea: number;
  maxFaceArea: number;
  minFillRatio: number;
  maxFillRatio: number;
  maxOffsetX: number;
  maxOffsetY: number;
};

export type OvalParams = {
  ovalWidthRatio: number;
  ovalHeightRatio: number;
  ovalCenterYRatio: number;
  ovalOffsetYRatio: number;
  minFillRatio: number;
  maxFillRatio: number;
  maxOffsetX: number;
  maxOffsetY: number;
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

export function genericOvalParams(closeUp: boolean): OvalParams {
  return {
    ovalWidthRatio: closeUp ? CAPTURE.widerOvalWidthRatio : CAPTURE.ovalWidthRatio,
    ovalHeightRatio: closeUp ? CAPTURE.widerOvalHeightRatio : CAPTURE.ovalHeightRatio,
    ovalCenterYRatio: CAPTURE.ovalCenterYRatio,
    ovalOffsetYRatio: CAPTURE.ovalOffsetYRatio,
    minFillRatio: CAPTURE.minFillRatio,
    maxFillRatio: CAPTURE.maxFillRatio,
    maxOffsetX: CAPTURE.maxOffsetX,
    maxOffsetY: CAPTURE.maxOffsetY,
  };
}

export function taskOvalParams(
  task: BankTask,
  size?: { widthRatio: number; heightRatio: number } & Partial<OvalParams>
): OvalParams {
  return {
    ovalWidthRatio: size?.widthRatio ?? task.ovalWidthRatio,
    ovalHeightRatio: size?.heightRatio ?? task.ovalHeightRatio,
    ovalCenterYRatio: task.ovalCenterYRatio ?? CAPTURE.ovalCenterYRatio,
    ovalOffsetYRatio: task.ovalOffsetYRatio ?? CAPTURE.ovalOffsetYRatio,
    minFillRatio: size?.minFillRatio ?? task.minFillRatio ?? CAPTURE.minFillRatio,
    maxFillRatio: size?.maxFillRatio ?? task.maxFillRatio ?? CAPTURE.maxFillRatio,
    maxOffsetX: size?.maxOffsetX ?? task.maxOffsetX ?? CAPTURE.maxOffsetX,
    maxOffsetY: size?.maxOffsetY ?? task.maxOffsetY ?? CAPTURE.maxOffsetY,
  };
}

export function computeOval(
  canvasWidth: number,
  canvasHeight: number,
  params: OvalParams
): OvalGeometry {
  const centerX = canvasWidth / 2;
  const centerY = canvasHeight / params.ovalCenterYRatio;
  const width = canvasWidth * params.ovalWidthRatio;
  const height = canvasHeight * params.ovalHeightRatio;
  const offsetCenterY = height * params.ovalOffsetYRatio;
  const area = Math.PI * (width / 2) * (height / 2);

  return {
    centerX,
    centerY,
    width,
    height,
    offsetCenterY,
    area,
    minFaceArea: params.minFillRatio * area,
    maxFaceArea: params.maxFillRatio * area,
    minFillRatio: params.minFillRatio,
    maxFillRatio: params.maxFillRatio,
    maxOffsetX: params.maxOffsetX,
    maxOffsetY: params.maxOffsetY,
  };
}

export function isFaceCentered(
  faceCenterX: number,
  faceCenterY: number,
  oval: OvalGeometry
) {
  const distanceX = Math.abs(faceCenterX - oval.centerX);
  const distanceY = Math.abs(faceCenterY - (oval.centerY + oval.offsetCenterY));
  return distanceX <= oval.maxOffsetX && distanceY <= oval.maxOffsetY;
}

export function isFaceWithinFillRange(
  faceWidth: number,
  faceHeight: number,
  oval: OvalGeometry
) {
  const faceArea = faceWidth * faceHeight;
  return faceArea >= oval.minFaceArea && faceArea <= oval.maxFaceArea;
}

export function guidanceText(opts: {
  fillPercent: number;
  faceCenterX: number;
  faceCenterY: number;
  oval: OvalGeometry;
  closeUp?: boolean;
  turnLabel?: string;
}): string {
  const { fillPercent, faceCenterX, faceCenterY, oval, closeUp, turnLabel } = opts;
  let instruction = "";

  if (fillPercent < oval.minFillRatio * 100) {
    instruction = "Move closer to the camera";
  } else if (fillPercent > oval.maxFillRatio * 100) {
    instruction = "Move further from the camera";
  } else {
    if (faceCenterX < oval.centerX - oval.maxOffsetX) {
      instruction = "Move right";
    } else if (faceCenterX > oval.centerX + oval.maxOffsetX) {
      instruction = "Move left";
    }
    if (faceCenterY < oval.centerY + oval.offsetCenterY - oval.maxOffsetY) {
      instruction += instruction ? " & down" : "Move down";
    } else if (faceCenterY > oval.centerY + oval.offsetCenterY + oval.maxOffsetY) {
      instruction += instruction ? " & up" : "Move up";
    }
  }

  if (instruction) return instruction;
  if (turnLabel === "Left") return "Turn your face left";
  if (turnLabel === "Right") return "Turn your face right";
  return closeUp
    ? "Position your face in the wider oval"
    : "Position your face in the oval";
}

export function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

export function sizeTransAt(task: BankTask, trans: OvalSizeTrans, t: number): OvalParams {
  const clamped = Math.max(0, Math.min(1, t));
  return taskOvalParams(task, {
    widthRatio: lerp(trans.fromWidthRatio, task.ovalWidthRatio, clamped),
    heightRatio: lerp(trans.fromHeightRatio, task.ovalHeightRatio, clamped),
    minFillRatio: lerp(trans.fromMinFillRatio, task.minFillRatio ?? CAPTURE.minFillRatio, clamped),
    maxFillRatio: lerp(trans.fromMaxFillRatio, task.maxFillRatio ?? CAPTURE.maxFillRatio, clamped),
    maxOffsetX: lerp(trans.fromMaxOffsetX, task.maxOffsetX ?? CAPTURE.maxOffsetX, clamped),
    maxOffsetY: lerp(trans.fromMaxOffsetY, task.maxOffsetY ?? CAPTURE.maxOffsetY, clamped),
  });
}
