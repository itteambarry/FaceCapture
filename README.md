# BB Face Capture

A full-screen face capture and liveness web app. It mirrors the flow of a camera-first KYC capture PWA: install prompt, brightness reminder, oval alignment, two-shot recording, optional color flash, and a ZIP download of photos and video.

## What it does

1. **Install overlay** — prompts to add FaceCapture to the home screen (Chrome install / iOS Add to Home Screen) or continue in the browser.
2. **Brightness guide** — asks you to turn the screen up before capture.
3. **Resolution picker** — 1920×1440, 1200×1600, 1080×1440, or 1280×720.
4. **Oval alignment** — MediaPipe face detection with live coaching (move closer, left, right, hold still).
5. **Generic capture** — `noflash` / `red` / `orange` run a two-shot oval (standard, then closer).
6. **Bank sequences** — pick `VPB`, `BIDV`, `VCB`, `DAB`, `VIB`, `VTB`, `KTB`, `STB`, or `SCB` to run that bank’s exact task list (oval size, flash colour, hold time, photo vs video, ZIP names).
   - **VCB / BIDV / VTB / STB** — NoFlash, then White / Red / Blue / Green holds.
   - **VIB** — Far, Near, Flash Grey, Flash Orange.
   - **DAB** — single standard take, 90° CW video, 1 Mbps.
   - **VPB** — Frontal, then turn Left, then turn Right (photos).
   - **SCB** — Far 10s, then Near with a small→large oval grow.
   - **KTB** — NoFlash, Green, Blue, Magenta, Gold with 90° CW 720p video.
7. **Download** — captures are packed into a ZIP in the browser.

Camera frames stay on the device. Nothing is uploaded.

## Run locally

```bash
npm install
npm run dev -- --port 43147 --hostname 127.0.0.1
```

Open [http://127.0.0.1:43147](http://127.0.0.1:43147). Allow camera access when the browser asks.

```bash
npm run build
npm start -- --port 43147
```

Production mode also registers a service worker so the app can be installed as a PWA.

## GitHub Pages

This repo deploys a static export to [https://itteambarry.github.io/FaceCapture/](https://itteambarry.github.io/FaceCapture/) on every push to `main`.

GitHub Pages can only serve static files, so the app is built with `output: "export"` and `basePath: /FaceCapture`. If the site still shows this README, switch **Settings → Pages → Source** to **GitHub Actions**.

## Notes

- HTTPS (or `localhost`) is required for camera and install.
- Face detection loads MediaPipe WASM from jsDelivr on first use.
- Face detection loads MediaPipe WASM from jsDelivr on first use.
- Video encoding matches the original PWA: raw camera H.264 MP4 at 16 Mbps when the browser allows it, otherwise WebM with the browser default bitrate. DAB is 1 Mbps rotated 1920×1440; KTB is 2 Mbps 720×1280 with no still photos. Bank JPEGs are quality 0.95 at that bank’s photo size.
- If the camera is blocked or missing, the app shows a permission error and a Restart control.
