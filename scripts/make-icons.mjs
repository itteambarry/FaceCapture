import { mkdir, writeFile } from "node:fs/promises";
import { deflateSync } from "node:zlib";

function crc32(buf) {
  let crc = -1;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ -1) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const t = Buffer.from(type);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([t, data])));
  return Buffer.concat([len, t, data, crc]);
}

function encodePng(width, height, getPixel) {
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    const row = (width * 4 + 1) * y;
    raw[row] = 0;
    for (let x = 0; x < width; x++) {
      const [r, g, b, a] = getPixel(x, y);
      const o = row + 1 + x * 4;
      raw[o] = r;
      raw[o + 1] = g;
      raw[o + 2] = b;
      raw[o + 3] = a;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function inRoundRect(x, y, size, radius) {
  const r = radius;
  if (x < 0 || y < 0 || x >= size || y >= size) return false;
  const cx = Math.min(Math.max(x, r), size - 1 - r);
  const cy = Math.min(Math.max(y, r), size - 1 - r);
  if (x === cx || y === cy) return true;
  return (x - cx) ** 2 + (y - cy) ** 2 <= r * r;
}

function inEllipse(x, y, cx, cy, rx, ry) {
  return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1;
}

function makeIcon(size) {
  const bg = [18, 18, 20, 255];
  const mint = [52, 211, 153, 255];
  const dark = [6, 40, 28, 255];
  const radius = size * 0.22;
  const cx = size / 2;
  const cy = size * 0.52;
  return encodePng(size, size, (x, y) => {
    if (!inRoundRect(x, y, size, radius)) return [0, 0, 0, 0];
    const face = inEllipse(x, y, cx, cy, size * 0.28, size * 0.36);
    const inner = inEllipse(x, y, cx, cy, size * 0.22, size * 0.29);
    const eyeL = inEllipse(x, y, cx - size * 0.1, cy - size * 0.06, size * 0.035, size * 0.04);
    const eyeR = inEllipse(x, y, cx + size * 0.1, cy - size * 0.06, size * 0.035, size * 0.04);
    if (eyeL || eyeR) return dark;
    if (face && !inner) return mint;
    return bg;
  });
}

await mkdir("public/icons", { recursive: true });
const sizes = {
  "public/icons/icon-512.png": 512,
  "public/icons/icon-512-maskable.png": 512,
  "public/icons/icon-192.png": 192,
  "public/apple-touch-icon.png": 180,
  "public/apple-touch-icon-180x180.png": 180,
  "public/apple-touch-icon-152x152.png": 152,
  "public/apple-touch-icon-120x120.png": 120,
  "public/icons/favicon-32.png": 32,
  "public/icons/favicon-16.png": 16,
};

for (const [path, size] of Object.entries(sizes)) {
  await writeFile(path, makeIcon(size));
  console.log("wrote", path);
}
