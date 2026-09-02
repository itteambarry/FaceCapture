"use client";

import { BrightnessGuide } from "@/components/brightness-guide";
import { CaptureStage } from "@/components/capture-stage";
import { InstallOverlay } from "@/components/install-overlay";
import { useIsStandalone } from "@/hooks/use-browser-flag";
import { useEffect, useState } from "react";

type Gate = "install" | "brightness" | "capture";

export function FaceCaptureApp() {
  const standalone = useIsStandalone();
  const [dismissedInstall, setDismissedInstall] = useState(false);
  const [dismissedBrightness, setDismissedBrightness] = useState(false);

  const gate: Gate = !standalone && !dismissedInstall
    ? "install"
    : !dismissedBrightness
      ? "brightness"
      : "capture";

  useEffect(() => {
    function preventScroll(event: TouchEvent) {
      const target = event.target as HTMLElement | null;
      if (!target) return;
      if (target.closest("select, input, textarea, button, a")) return;
      if (event.touches.length > 1) return;
      event.preventDefault();
    }
    document.addEventListener("touchmove", preventScroll, { passive: false });
    return () => document.removeEventListener("touchmove", preventScroll);
  }, []);

  return (
    <>
      <CaptureStage active={gate === "capture"} />
      <InstallOverlay
        open={gate === "install"}
        onContinue={() => setDismissedInstall(true)}
      />
      <BrightnessGuide
        open={gate === "brightness"}
        onDone={() => setDismissedBrightness(true)}
      />
    </>
  );
}
