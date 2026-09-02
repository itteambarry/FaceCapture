"use client";

/* Overlay uses static PNGs; next/image is unnecessary here. */
/* eslint-disable @next/next/no-img-element */

import { Button } from "@/components/ui/button";
import { withBase } from "@/lib/base-path";
import { useBrowserFlag } from "@/hooks/use-browser-flag";
import { APP_NAME, APP_SHORT_NAME } from "@/lib/capture-config";
import {
  isGoogleChrome,
  isInstallCompatible,
  isIosDevice,
  isSafariBrowser,
  isSecureInstallContext,
} from "@/lib/pwa";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

type BeforeInstallPromptEvent = Event & {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
};

type InstallOverlayProps = {
  open: boolean;
  onContinue: () => void;
};

export function InstallOverlay({ open, onContinue }: InstallOverlayProps) {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(
    null
  );
  const [waited, setWaited] = useState(false);
  const ios = useBrowserFlag(isIosDevice);
  const safari = useBrowserFlag(isSafariBrowser);
  const chrome = useBrowserFlag(isGoogleChrome);
  const compatible = useBrowserFlag(isInstallCompatible);
  const sslOk = useBrowserFlag(isSecureInstallContext, true);

  useEffect(() => {
    const onPrompt = (event: Event) => {
      event.preventDefault();
      setDeferredPrompt(event as BeforeInstallPromptEvent);
    };
    window.addEventListener("beforeinstallprompt", onPrompt);
    const timer = window.setTimeout(() => setWaited(true), 1800);
    return () => {
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.clearTimeout(timer);
    };
  }, []);

  const statusOk = sslOk && compatible;
  const statusText = statusOk
    ? "Installation Compatible : Yes"
    : "Installation Compatible : No";

  const showIosGuide = safari && ios;
  const showDrawerGuide = chrome && waited && !deferredPrompt && compatible;

  async function handleInstall() {
    if (deferredPrompt) {
      await deferredPrompt.prompt();
      const choice = await deferredPrompt.userChoice;
      if (choice.outcome === "accepted") {
        setDeferredPrompt(null);
      }
      return;
    }
    onContinue();
  }

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-5 pb-[max(20px,env(safe-area-inset-bottom))]"
      role="dialog"
      aria-modal="true"
      aria-label="Install app"
    >
      <div className="w-full max-w-[400px] rounded-2xl border border-emerald-400/45 bg-[#141416] p-5 pt-[22px] shadow-[0_12px_40px_rgba(0,0,0,.5)]">
        <img
          src={withBase("/icons/icon-192.png")}
          alt=""
          width={56}
          height={56}
          className="mx-auto mb-3 block size-14 rounded-xl bg-[#1a1a1a]"
        />
        <h1 className="mb-1.5 text-center text-lg font-bold text-white">
          Install {APP_NAME}
        </h1>
        <p className="mb-3 text-center text-[13px] leading-snug text-white/70">
          Install {APP_NAME} for better capture result.
        </p>
        <div
          className={cn(
            "mb-3 rounded-[10px] bg-white/6 px-3 py-2.5 text-center text-[13px] leading-snug",
            statusOk
              ? "border border-emerald-400/35 text-emerald-200"
              : "border border-red-400/45 text-red-200"
          )}
        >
          {statusText}
        </div>

        {showDrawerGuide ? (
          <div className="mb-3">
            <div className="mx-auto w-[min(220px,70%)] rounded-[22px] border-2 border-white/20 bg-[#0a0a0c] px-2.5 pb-3 pt-2.5">
              <div className="mx-auto mb-2 h-1.5 w-[36%] rounded bg-white/12" />
              <div className="relative h-[200px] overflow-hidden rounded-xl bg-linear-to-b from-[#1a1b22] to-[#0e0f14]">
                <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_30%_25%,rgba(52,211,153,.35),transparent_45%),radial-gradient(circle_at_80%_60%,rgba(59,130,246,.25),transparent_40%)] opacity-35" />
                <img
                  src={withBase("/icons/icon-192.png")}
                  alt=""
                  className="absolute top-[28%] left-1/2 z-10 size-[52px] -translate-x-1/2 rounded-xl shadow-lg animate-[flyDrawer_2.4s_ease-in-out_infinite]"
                />
                <div className="absolute top-[42%] left-1/2 z-20 ml-2.5 text-[22px] drop-shadow animate-[handTap_2.4s_ease-in-out_infinite]">
                  👆
                </div>
                <div className="absolute right-2 bottom-2 left-2 z-10 flex h-16 flex-col items-center justify-end rounded-xl border border-emerald-400/35 bg-white/8 pb-1.5">
                  <div className="mb-1.5 text-[10px] tracking-wide text-emerald-200 uppercase">
                    App Drawer
                  </div>
                  <div className="flex items-center justify-center gap-2">
                    <span className="size-[22px] rounded-md bg-white/12" />
                    <span className="size-7 rounded-lg border border-dashed border-emerald-400/70 bg-emerald-400/20" />
                    <span className="size-[22px] rounded-md bg-white/12" />
                    <span className="size-[22px] rounded-md bg-white/12" />
                  </div>
                </div>
              </div>
            </div>
            <p className="mt-2.5 text-center text-[13px] font-semibold text-emerald-200">
              Open the {APP_SHORT_NAME} app from App Drawer
            </p>
          </div>
        ) : null}

        {showIosGuide ? (
          <div className="mb-3">
            <div className="mx-auto w-[min(220px,70%)] rounded-[22px] border-2 border-white/20 bg-[#0a0a0c] px-2.5 pb-3 pt-2.5">
              <div className="mx-auto mb-2 h-1.5 w-[36%] rounded bg-white/12" />
              <div className="relative h-[220px] overflow-hidden rounded-xl bg-[#111111]">
                <div className="flex flex-col items-center pt-6">
                  <img
                    src={withBase("/icons/icon-192.png")}
                    alt=""
                    className="size-14 rounded-xl"
                  />
                  <div className="mt-1.5 text-xs font-semibold text-white">
                    {APP_SHORT_NAME}
                  </div>
                </div>
                <div className="absolute right-2 bottom-9 left-2 rounded-xl bg-[#2c2c2e] py-1 text-[12px] text-white">
                  <div className="px-3 py-1.5 opacity-60">Copy</div>
                  <div className="flex items-center gap-2 bg-emerald-400/15 px-3 py-1.5 font-semibold text-emerald-200">
                    <span>＋</span>
                    <span>Add to Home Screen</span>
                  </div>
                  <div className="px-3 py-1.5 opacity-60">Find on Page</div>
                </div>
                <div className="absolute right-3 bottom-2 left-3 flex justify-between text-white/80">
                  <span>‹</span>
                  <span>›</span>
                  <ShareGlyph />
                  <span>□</span>
                </div>
                <div className="absolute right-8 bottom-1 animate-[handTap_2.4s_ease-in-out_infinite] text-xl">
                  👆
                </div>
              </div>
            </div>
            <p className="mt-2.5 text-center text-[13px] font-semibold text-emerald-200">
              Tap Share → Add to Home Screen
            </p>
          </div>
        ) : null}

        <div className="flex gap-2">
          <Button
            type="button"
            variant="secondary"
            className="h-11 flex-1 rounded-[10px] bg-white/10 text-sm font-bold text-white hover:bg-white/16"
            onClick={onContinue}
          >
            Continue in browser
          </Button>
          <Button
            type="button"
            className="h-11 flex-1 rounded-[10px] bg-[#34D399] text-sm font-bold text-[#06281c] hover:bg-[#34D399]/90 disabled:opacity-45"
            disabled={!compatible && !safari}
            onClick={handleInstall}
          >
            {safari
              ? "Got it"
              : deferredPrompt
                ? "Install app"
                : waited
                  ? "Retry"
                  : "Waiting…"}
          </Button>
        </div>
        <p className="mt-2.5 text-center text-[11px] leading-snug text-white/45">
          {safari
            ? "After installation, open FaceCapture from Home Screen."
            : "You can Continue in browser anytime."}
        </p>
      </div>
    </div>
  );
}

function ShareGlyph() {
  return (
    <svg viewBox="0 0 24 24" className="inline-block size-4 text-blue-400" aria-hidden>
      <path
        d="M12 3v11"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M8 6.5L12 3l4 3.5"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <path
        d="M6 12v6.5A1.5 1.5 0 0 0 7.5 20h9a1.5 1.5 0 0 0 1.5-1.5V12"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}
