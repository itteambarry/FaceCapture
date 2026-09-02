export function isStandaloneDisplay() {
  if (typeof window === "undefined") return false;
  return (
    window.matchMedia("(display-mode: standalone)").matches ||
    window.matchMedia("(display-mode: fullscreen)").matches ||
    window.matchMedia("(display-mode: minimal-ui)").matches ||
    Boolean((window.navigator as Navigator & { standalone?: boolean }).standalone)
  );
}

export function isIosDevice() {
  if (typeof navigator === "undefined") return false;
  return (
    /iphone|ipad|ipod/i.test(navigator.userAgent) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1)
  );
}

export function isSafariBrowser() {
  if (typeof navigator === "undefined") return false;
  if (isIosDevice()) return true;
  const ua = navigator.userAgent;
  if (!/Safari\//i.test(ua)) return false;
  if (/Chrome\/|Chromium\/|Edg\/|OPR\/|Opera|Firefox\//i.test(ua)) return false;
  return true;
}

export function isGoogleChrome() {
  if (typeof navigator === "undefined" || typeof window === "undefined") return false;
  if (isIosDevice()) return false;
  const ua = navigator.userAgent;
  if (!("chrome" in window)) return false;
  if (!/Chrome\/\d+/i.test(ua)) return false;
  if (/Edg\/|OPR\/|Opera|SamsungBrowser|Firefox\//i.test(ua)) return false;
  if (/; wv\)/i.test(ua)) return false;
  return true;
}

export function isSecureInstallContext() {
  if (typeof window === "undefined") return false;
  const host = window.location.hostname;
  if (host === "localhost" || host === "127.0.0.1") {
    return window.location.protocol === "https:" || window.location.protocol === "http:";
  }
  return window.isSecureContext === true && window.location.protocol === "https:";
}

export function isInstallCompatible() {
  if (!isSecureInstallContext()) return false;
  if (isSafariBrowser()) return true;
  if (isGoogleChrome()) return "serviceWorker" in navigator;
  return false;
}
