"use client";

import { withBase } from "@/lib/base-path";
import { useEffect } from "react";

export function ServiceWorkerRegister() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production") return;
    if (!("serviceWorker" in navigator)) return;
    void navigator.serviceWorker.register(withBase("/sw.js"));
  }, []);
  return null;
}
