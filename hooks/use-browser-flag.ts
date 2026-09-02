"use client";

import { isStandaloneDisplay } from "@/lib/pwa";
import { useSyncExternalStore } from "react";

function emptySubscribe() {
  return () => {};
}

export function useBrowserFlag(read: () => boolean, serverValue = false) {
  return useSyncExternalStore(emptySubscribe, read, () => serverValue);
}

export function useIsStandalone() {
  return useBrowserFlag(isStandaloneDisplay);
}
