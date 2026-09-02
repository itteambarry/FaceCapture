# BB Face Capture

A full-screen face capture and liveness web app. It mirrors the flow of a camera-first KYC capture PWA: install prompt, brightness reminder, oval alignment, two-shot recording, optional color flash, and a ZIP download of photos and video.

## What it does

1. **Install overlay** — prompts to add FaceCapture to the home screen (Chrome install / iOS Add to Home Screen) or continue in the browser.
2. **Brightness guide** — asks you to turn the screen up before capture.
3. **Resolution picker** — 1920×1440, 1200×1600, 1080×1440, or 1280×720.
4. **Oval alignment** — MediaPipe face detection with live coaching (move closer, left, right, hold still).
5. **Two captures** — a standard oval, then a closer / wider oval. Each records ~5 seconds of video plus a JPEG.
6. **Flash modes** — `noflash`, `red`, `orange`, `white`, `blue`, `green`.
7. **Download** — both takes are packed into a ZIP in the browser.

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

## Notes

- HTTPS (or `localhost`) is required for camera and install.
- Face detection loads MediaPipe WASM from jsDelivr on first use.
- If the camera is blocked or missing, the app shows a permission error and a Restart control.
