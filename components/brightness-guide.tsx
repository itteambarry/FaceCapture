"use client";

import { Button } from "@/components/ui/button";

type BrightnessGuideProps = {
  open: boolean;
  onDone: () => void;
};

export function BrightnessGuide({ open, onDone }: BrightnessGuideProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/82 p-5 pb-[max(20px,env(safe-area-inset-bottom))]"
      role="dialog"
      aria-modal="true"
      aria-label="Brightness Settings"
    >
      <div className="w-full max-w-[400px] rounded-2xl border border-yellow-400/45 bg-[#141416] p-5 pt-[22px] shadow-[0_12px_40px_rgba(0,0,0,.5)]">
        <h1 className="mb-2 text-center text-lg font-bold text-white">
          Brightness Settings
        </h1>
        <p className="mb-4 text-center text-sm leading-snug text-white/80">
          Please maximize the screen brightness.
        </p>
        <div className="mx-auto mb-4 w-[min(240px,80%)] rounded-[18px] border-2 border-white/14 bg-[#0a0a0c] p-4">
          <div className="relative flex h-40 flex-col items-center justify-center gap-3.5 overflow-hidden rounded-xl bg-[#1a1a1a]">
            <div className="pointer-events-none absolute inset-0 animate-[brightnessGlow_2.2s_ease-in-out_infinite] bg-[radial-gradient(circle_at_50%_40%,rgba(250,204,21,.55),transparent_65%)]" />
            <div className="relative z-10 size-14 animate-[sunPulse_2.2s_ease-in-out_infinite] rounded-full bg-[radial-gradient(circle_at_35%_35%,#fde68a,#f59e0b_55%,#d97706)] shadow-[0_0_24px_rgba(250,204,21,.65)]" />
            <div className="relative z-10 h-2.5 w-[78%] overflow-hidden rounded-full bg-white/12">
              <div className="h-full w-[92%] rounded-full bg-yellow-400" />
            </div>
            <div className="relative z-10 flex w-[78%] justify-between text-[10px] text-white/55">
              <span>Low</span>
              <span>Max</span>
            </div>
          </div>
        </div>
        <p className="mb-4 text-center text-[13px] font-semibold text-yellow-200">
          Drag the system brightness slider all the way up
        </p>
        <Button
          type="button"
          className="h-11 w-full rounded-[10px] bg-[#34D399] text-sm font-bold text-[#06281c] hover:bg-[#34D399]/90"
          onClick={onDone}
        >
          Done
        </Button>
      </div>
    </div>
  );
}
