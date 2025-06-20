import { FilesetResolver, FaceDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

// 初始設定
const config = {
  confidenceThreshold: 0.8, //人臉最低概率
  maxOffsetX: 25, //人臉左右最大偏移值
  maxOffsetY: 30, //人臉上下最大偏移值
  minFillRatio: 0.85, //人臉識別框占識別區域的最小佔比
  maxFillRatio: 1.1, //人臉識別框占識別區域的最大佔比
  ovalWidthRatio: 0.5, //識別區域寬度
  ovalHeightRatio: 0.6, //識別區域長度
  widerOvalWidthRatio: 0.65, //近距離識別區域寬度
  widerOvalHeightRatio: 0.75, //近距離識別區域長度
  msbOvalWidthRatio: 0.45, //MSB_Flash識別區域寬度 (slightly smaller)
  msbOvalHeightRatio: 0.55, //MSB_Flash識別區域長度 (slightly smaller)
  ovalCenterYRatio: 2.0, //識別區域中心點在canvas中y軸偏移比例
  ovalOffsetYRatio: 0.125, //識別區域內中心點y軸偏移比例
  countdownDuration: 5,  //臉部錄像時長
  flashSecond: 0.8, //閃屏間隔時長
  flashFaceCaptureSec: 5,
  availableOptions: ["noflash","red","orange", "VIB_Flash", "MSB_Flash"] //閃屏顏色設置
};

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const instructionText = document.getElementById("instruction");
const progressBarContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");
const restartButton = document.getElementById("restartBtn");
const flashBlock = document.getElementById("flashBlock");
const colorSelect = document.getElementById("colorSelect");

let faceDetector;
let canvasWidth, canvasHeight;
let ovalCenterX, ovalCenterY, ovalWidth, ovalHeight, offsetOvalCenterY, ovalArea;
let minFaceArea, maxFaceArea;
let isValidFace = false;
let animationFrameId;
let lastDetectionTime = 0;
const FPS_TARGET = 30; //臉部偵測幀數
const FRAME_INTERVAL = 1000 / FPS_TARGET;

let isSecondCapture = false; //secondcapture 為近距離臉部錄像
let isFlashFaceCapture = false;
let isMSBFlashCapture = false; //MSB_Flash錄像
let primaryBlobs = {
  video: null,
  photo: null
};
let secondaryBlobs = {
  video: null,
  photo: null
};
let flashFaceBlobs = {
  video: null,
  photo: null
}
let msbFlashBlobs = {
  video: null,
  photo: null
}

let countdownActive = false;
let countdownStartTime = 0;
let mediaRecorder = null;
let recordedChunks = [];
let detectionActive = true; 
let photoCanvas = document.createElement('canvas'); 
let photoCtx = photoCanvas.getContext('2d');

function resetApplication() {
    countdownActive = false;
    isValidFace = false;
    isSecondCapture = false;
    isFlashFaceCapture = false;
    isMSBFlashCapture = false;

    recordedChunks = [];
    primaryBlobs = {
      video: null,
      photo: null
    };
    secondaryBlobs = {
      video: null,
      photo: null
    };
    flashFaceBlobs = {
      video: null,
      photo: null
    }
    msbFlashBlobs = {
      video: null,
      photo: null
    }
    
    updateOvalDimensions();

    restartButton.style.display = "none";
    progressBarContainer.style.display = "none";
    progressBar.style.width = "0%";
    instructionText.innerText = "Position your face in the oval";

    detectionActive = true;
    detectFaces();
}

function initOptions(){
    config.availableOptions.forEach(color => {
        if(color === "VIB_Flash")
        {
          var option = document.createElement("option");
          option.value = "#ee5d35";
          option.innerText = "VIB_Flash";
          colorSelect.appendChild(option);
        }
        else{
          var option = document.createElement("option");
          option.value = color;
          option.innerText = color;
          colorSelect.appendChild(option);
        }
    });
    colorSelect.selectedIndex = 0;
    colorSelect.hidden = false;
}

function setColor() {
    if(colorSelect.value == config.availableOptions[0])
    {
        flashBlock.style.opacity = 0;
        colorSelect.style.backgroundColor = "grey";
    }
    else
    {
        flashBlock.style.opacity = 1;
        flashBlock.style.backgroundColor = colorSelect.value;
        colorSelect.style.backgroundColor = colorSelect.value;
    }
}

async function initialize() {
  try {
    await setupCamera();
    resizeCanvas();
    await loadFaceDetector();
    video.play();
    detectFaces();

    initOptions();

    restartButton.onclick = resetApplication;
    colorSelect.onchange = setColor;
    
    let resizeTimeout;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        resizeCanvas();
      }, 250);
    });
  } catch (error) {
    console.error("Initialization error:", error);
    instructionText.innerText = "Camera access error. Please reload and allow camera permissions.";
  }
}

async function setupCamera() {
  const isPortrait = window.matchMedia("(orientation: portrait)").matches;
  const aspectRatio = isPortrait ? 4/3 : 3/4;
  
  const videoSetting = { 
    video: {
      width: { ideal: 1200 },
      height: { ideal: 1600 },
      aspectRatio: { exact: aspectRatio }
    }, 
    facingMode: "user" 
  };
  
  const stream = await navigator.mediaDevices.getUserMedia(videoSetting);
  video.srcObject = stream;

  return new Promise(resolve => {
    video.onloadedmetadata = () => resolve(video);
  });
}

async function loadFaceDetector() {
  try {
    const visionFileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );

    faceDetector = await FaceDetector.createFromOptions(visionFileset, {
      baseOptions: { 
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        delegate: "GPU"
      },
      runningMode: "VIDEO"
    });
    
  } catch (error) {
    const visionFileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );

    faceDetector = await FaceDetector.createFromOptions(visionFileset, {
      baseOptions: { 
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
      },
      runningMode: "VIDEO"
    });
  }
}

function resizeCanvas() {
  const aspectRatio = 3/4;
  const padding = 20;
  const maxWidth = window.innerWidth - (padding * 2);
  const maxHeight = window.innerHeight - (padding * 2);
  
  let width, height;
  
  if (maxWidth / aspectRatio <= maxHeight) {
    width = maxWidth;
    height = width / aspectRatio;
  } else {
    height = maxHeight;
    width = height * aspectRatio;
  }
  
  canvas.width = Math.floor(width);
  canvas.height = Math.floor(height);
  canvasWidth = canvas.width;
  canvasHeight = canvas.height;

  updateOvalDimensions();
}

function updateOvalDimensions() {
  ovalCenterX = canvasWidth / 2;
  ovalCenterY = canvasHeight / config.ovalCenterYRatio;
  
  // Check if MSB_Flash is selected for smaller standard oval
  const isMSBFlashSelected = colorSelect.options[colorSelect.selectedIndex]?.innerText === "MSB_Flash";
  
  if (!isSecondCapture) {
    if (isMSBFlashSelected) {
      ovalWidth = canvasWidth * config.msbOvalWidthRatio;
      ovalHeight = canvasHeight * config.msbOvalHeightRatio;
    } else {
      ovalWidth = canvasWidth * config.ovalWidthRatio;
      ovalHeight = canvasHeight * config.ovalHeightRatio;
    }
  } else {
    ovalWidth = canvasWidth * config.widerOvalWidthRatio;
    ovalHeight = canvasHeight * config.widerOvalHeightRatio;
  }
  
  offsetOvalCenterY = ovalHeight * config.ovalOffsetYRatio;
  ovalArea = Math.PI * (ovalWidth / 2) * (ovalHeight / 2);

  minFaceArea = config.minFillRatio * ovalArea;
  maxFaceArea = config.maxFillRatio * ovalArea;
}

async function detectFaces() {
  if (!faceDetector || !detectionActive) return;
  
  const now = performance.now();
  const elapsed = now - lastDetectionTime;
  
  if (elapsed >= FRAME_INTERVAL) {
    lastDetectionTime = now;
    
    try {
      const detections = await faceDetector.detectForVideo(video, now);
      renderFrame(detections);
      handleCountdown(now);
    } catch (error) {
      console.error("Detection error:", error);
    }
  }
  
  animationFrameId = requestAnimationFrame(detectFaces);
}

function handleCountdown(now) {
  if (isValidFace) {
    if (!countdownActive) {
      countdownActive = true;
      countdownStartTime = now;
      progressBarContainer.style.display = "block";
      progressBar.style.width = "0%";
      
      startRecording();
    } else {
      const elapsedSeconds = (now - countdownStartTime) / 1000;
      const progressPercent = Math.min((elapsedSeconds / config.countdownDuration) * 100, 100);

      if((elapsedSeconds > config.countdownDuration / 4) & colorSelect.options[colorSelect.selectedIndex].innerText != "VIB_Flash" && colorSelect.options[colorSelect.selectedIndex].innerText != "MSB_Flash") {
        flashBlock.hidden = elapsedSeconds % config.flashSecond * 2 < config.flashSecond;
      }
      else if(isFlashFaceCapture || isMSBFlashCapture)
      {
        flashBlock.hidden = false;
      }

      progressBar.style.width = `${progressPercent}%`;
      
      if (elapsedSeconds >= config.countdownDuration) {
        completeCountdown();
      }
    }
  } else if (countdownActive) {
    cancelCountdown();
  }
}

function startRecording() {
  if (mediaRecorder) {
    stopRecording();
  }
  
  recordedChunks = [];
  const stream = video.srcObject;
  
  try {
    const mimeType = "video/mp4;codecs=avc1.42E01E, mp4a.40.2";
    const options = {
      mimeType: mimeType,
      videoBitsPerSecond: 16000000 //bitrate 越高畫質越好
    };
    
    try {
      mediaRecorder = new MediaRecorder(stream, options);
    } catch (e) {
      console.warn("MP4 recording not supported, falling back to WebM", e);
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    }
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };
    
    mediaRecorder.start();
  } catch (error) {
    console.error("Error starting MediaRecorder:", error);
    instructionText.innerText = "Recording not supported in this browser";
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
}

function capturePhoto() {
  photoCanvas.width = video.videoWidth;
  photoCanvas.height = video.videoHeight;

  photoCtx.drawImage(video, 0, 0, photoCanvas.width, photoCanvas.height);
  
  return new Promise((resolve) => {
    photoCanvas.toBlob((blob) => {
      resolve(blob);
    }, 'image/jpeg', 0.95); //0-1 畫質
  });
}

async function processPrimaryCapture() {
  if (recordedChunks.length === 0) {
    console.log("No data recorded in primary capture");
    return false;
  }
  
  try {
    const mimeType = mediaRecorder.mimeType;
    primaryBlobs.video = new Blob(recordedChunks, { type: mimeType });
    primaryBlobs.photo = await capturePhoto();
    
    return true;
  } catch (error) {
    console.error("Error processing primary capture:", error);
    return false;
  }
}

function startSecondaryCapture() {
  isSecondCapture = true;
  
  updateOvalDimensions();
  
  countdownActive = false;
  isValidFace = false;
  recordedChunks = [];
  
  instructionText.innerText = "Position for second capture (larger frame)";
  
  progressBarContainer.style.display = "none";
  progressBar.style.width = "0%";
  
  detectionActive = true;
  detectFaces();
}

async function processSecondaryCapture() {
  if (recordedChunks.length === 0) {
    console.log("No data recorded in secondary capture");
    return false;
  }
  
  try {
    const mimeType = mediaRecorder.mimeType;
    secondaryBlobs.video = new Blob(recordedChunks, { type: mimeType });
    secondaryBlobs.photo = await capturePhoto();
    
    return true;
  } catch (error) {
    console.error("Error processing secondary capture:", error);
    return false;
  }
}

function startFlashFaceCapture() {
  isFlashFaceCapture = true;
  
  countdownActive = false;
  isValidFace = false;
  recordedChunks = [];
  
  instructionText.innerText = "Position for Flash Capture (Light Reflection Frame)";
  
  progressBarContainer.style.display = "none";
  progressBar.style.width = "0%";
  
  detectionActive = true;
  detectFaces();
}

async function saveAllCaptures() {
  try {
    const timestamp = new Date().toISOString().replace(/:/g, '-');
    
    const mimeType = mediaRecorder.mimeType;
    const fileExtension = mimeType.includes('mp4') ? 'mp4' : 'webm';
    
    const zip = new JSZip();
    
    zip.file(`face-capture-standard-${timestamp}.${fileExtension}`, primaryBlobs.video);
    zip.file(`face-capture-standard-${timestamp}.jpg`, primaryBlobs.photo);
    
    zip.file(`face-capture-wide-${timestamp}.${fileExtension}`, secondaryBlobs.video);
    zip.file(`face-capture-wide-${timestamp}.jpg`, secondaryBlobs.photo);

    if(colorSelect.options[colorSelect.selectedIndex].innerText == "VIB_Flash") {
      zip.file(`face-capture-VIB_Flash-${timestamp}.${fileExtension}`, flashFaceBlobs.video);
      zip.file(`face-capture-VIB_Flash-${timestamp}.jpg`, flashFaceBlobs.photo);
    }
    
    if(colorSelect.options[colorSelect.selectedIndex].innerText == "MSB_Flash") {
      zip.file(`face-capture-MSB_Flash-${timestamp}.${fileExtension}`, msbFlashBlobs.video);
      zip.file(`face-capture-MSB_Flash-${timestamp}.jpg`, msbFlashBlobs.photo);
    }
    
    const zipBlob = await zip.generateAsync({type: "blob"});
    const zipUrl = URL.createObjectURL(zipBlob);
    const a = document.createElement('a');
    
    a.style.display = 'none';
    a.href = zipUrl;
    a.download = `face-captures-${timestamp}.zip`;
    
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
      document.body.removeChild(a);
      window.URL.revokeObjectURL(zipUrl);
    }, 100);
    
    return true;
  } catch (error) {
    console.error("Error saving files:", error);
    instructionText.innerText = "Error saving files";
    return false;
  }
}

async function completeSecondCapture() {
  countdownActive = false;
  progressBarContainer.style.display = "none";
  flashBlock.hidden = true;
  
  detectionActive = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
    
  instructionText.innerText = "Processing captures...";
      
      try {
        secondaryBlobs.video = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
        secondaryBlobs.photo = await capturePhoto();
        
        if(colorSelect.options[colorSelect.selectedIndex].innerText != "VIB_Flash" && colorSelect.options[colorSelect.selectedIndex].innerText != "MSB_Flash") {
          const saveSuccess = await saveAllCaptures();
          if (saveSuccess) {
            instructionText.innerText = "All captures saved!";
          } else {
            instructionText.innerText = "Error saving files";
          }
        }
      } catch (error) {
        console.error("Error processing second capture:", error);
        instructionText.innerText = "Error processing captures";
      }

      restartButton.style.display = "block";
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
}

async function completeFlashFaceCapture() {
  countdownActive = false;
  progressBarContainer.style.display = "none";
  flashBlock.hidden = true;
  
  detectionActive = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
    
  instructionText.innerText = "Processing captures...";
      
      try {
        flashFaceBlobs.video = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
        flashFaceBlobs.photo = await capturePhoto();
        
        const saveSuccess = await saveAllCaptures();
        
        if (saveSuccess) {
          instructionText.innerText = "All captures saved!";
        } else {
          instructionText.innerText = "Error saving files";
        }
      } catch (error) {
        console.error("Error processing second capture:", error);
        instructionText.innerText = "Error processing captures";
      }
      
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }

      restartButton.style.display = "block";
}

async function completeCountdown() {
  countdownActive = false;
  detectionActive = false;
  progressBarContainer.style.display = "none";
  flashBlock.hidden = true;
  
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    
    mediaRecorder.onstop = async () => {
      if (!isSecondCapture) {
        instructionText.innerText = "Processing first capture...";
        const primarySuccess = await processPrimaryCapture();
        
        if (primarySuccess) {
          startSecondaryCapture();
        } else {
          instructionText.innerText = "Error processing first capture";
          restartButton.style.display = "block";
        }
      } else if(isSecondCapture && !isFlashFaceCapture && !isMSBFlashCapture) {
        if(colorSelect.options[colorSelect.selectedIndex].innerText == "VIB_Flash")
        {
          const secondarySuccess = await processSecondaryCapture();
          if(secondarySuccess) {
            startFlashFaceCapture();
          }
        }
        else{
          completeSecondCapture();
        }
      }
      else{
        completeFlashFaceCapture();
      }
    };
  }
}

function cancelCountdown() {
  countdownActive = false;
  progressBarContainer.style.display = "none";
  progressBar.style.width = "0%";
  flashBlock.hidden = true;
  stopRecording();
  
  instructionText.innerText = "Please keep your face in position";
  
  setTimeout(() => {
    if (!isValidFace) {
      if (isSecondCapture) {
        instructionText.innerText = "Position for second capture";
      } else {
        instructionText.innerText = "Position your face in the oval";
      }
    }
  }, 2000);
}

function renderFrame(detections) {
  const ratio = canvas.width / video.videoWidth;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  
  isValidFace = false;
  
  createOvalMask();
  
  if (detections.detections.length > 0) {
    const detection = detections.detections.sort((a, b) => 
      b.categories[0].score - a.categories[0].score
    )[0];
    
    processFaceDetection(detection, ratio);
  } else {
    if (!countdownActive) {
      if (isSecondCapture) {
        instructionText.innerText = "No face detected for second capture";
      } else {
        instructionText.innerText = "No face detected";
      }
    }
  }
  
  drawOvalBorder();
  applyBackgroundMask();
}

function createOvalMask() {
  ctx.globalCompositeOperation = "destination-in";
  ctx.beginPath();
  ctx.ellipse(ovalCenterX, ovalCenterY, ovalWidth / 2, ovalHeight / 2, 0, 0, 2 * Math.PI);
  ctx.fillStyle = "white";
  ctx.fill();
  ctx.globalCompositeOperation = "source-over";
}

function processFaceDetection(detection, ratio) {
  const { boundingBox, categories } = detection;
  const confidence = categories[0].score;
  
  if (confidence < config.confidenceThreshold) {
    if (!countdownActive) {
      if (isSecondCapture) {
        instructionText.innerText = "No face detected for second capture";
      } else {
        instructionText.innerText = "No face detected";
      }
    }
    return;
  }
  
  let x = boundingBox.originX * ratio;
  let y = boundingBox.originY * ratio;
  let width = boundingBox.width * ratio;
  let height = boundingBox.height * ratio;
  
  x = canvas.width - (x + width);
  
  const faceCenterX = x + width / 2;
  const faceCenterY = y + height / 2;
  const fillPercent = ((width * height) / ovalArea) * 100;
  
  const withinFillRange = isFaceWithinFillRange(width, height);
  const isCentered = isFaceCentered(faceCenterX, faceCenterY);
  
  isValidFace = withinFillRange && isCentered;
  
  if (!countdownActive) {
    updateInstruction(fillPercent, faceCenterX, faceCenterY);
  }
  
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    displayMetrics(x, y, width, height, confidence, fillPercent, faceCenterX, faceCenterY);
  }
}

function displayMetrics(x, y, width, height, confidence, fillPercent, faceCenterX, faceCenterY) {
  const offsetX = Math.abs(faceCenterX - ovalCenterX);
  const offsetY = Math.abs(faceCenterY - (ovalCenterY + offsetOvalCenterY));
  
  ctx.strokeStyle = isValidFace ? "green" : "red";
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, width, height);
  ctx.fillStyle = "white";
  ctx.font = "16px Arial";
  ctx.fillText(`Confidence: ${(confidence * 100).toFixed(1)}%`, x, y - 20);
  ctx.fillText(`Fill: ${fillPercent.toFixed(1)}%`, x, y);
  ctx.fillText(`OffsetX: ${offsetX.toFixed(1)} px`, x, y + height + 20);
  ctx.fillText(`OffsetY: ${offsetY.toFixed(1)} px`, x, y + height + 40);

  ctx.fillText(`Mode: ${isSecondCapture ? "Wide" : "Standard"}`, x, y - 40);
}

function drawOvalBorder() {
  ctx.beginPath();
  ctx.ellipse(ovalCenterX, ovalCenterY, ovalWidth / 2, ovalHeight / 2, 0, 0, 2 * Math.PI);
  ctx.strokeStyle = isValidFace ? "green" : "red";
  ctx.lineWidth = 3;
  ctx.stroke();
}

function applyBackgroundMask() {
  ctx.globalCompositeOperation = "destination-over";
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "source-over";
}

function isFaceWithinFillRange(width, height) {
  const faceArea = width * height;
  return faceArea >= minFaceArea && faceArea <= maxFaceArea;
}

function isFaceCentered(faceCenterX, faceCenterY) {
  const distanceX = Math.abs(faceCenterX - ovalCenterX);
  const distanceY = Math.abs(faceCenterY - (ovalCenterY + offsetOvalCenterY));
  return distanceX <= config.maxOffsetX && distanceY <= config.maxOffsetY;
}

function updateInstruction(fillPercent, faceCenterX, faceCenterY) {
  if (isValidFace) {
    instructionText.innerText = "Hold still for "+config.countdownDuration+" seconds";
    return;
  }
  
  let instruction = "";
  
  if (fillPercent < config.minFillRatio * 100) {
    instruction = "Move closer to the camera";
  } else if (fillPercent > config.maxFillRatio * 100) {
    instruction = "Move further from the camera";
  } else {
    if (faceCenterX < ovalCenterX - config.maxOffsetX) {
      instruction = "Move right";
    } else if (faceCenterX > ovalCenterX + config.maxOffsetX) {
      instruction = "Move left";
    }
    if (faceCenterY < (ovalCenterY + offsetOvalCenterY) - config.maxOffsetY) {
      instruction += instruction ? " & down" : "Move down";
    } else if (faceCenterY > (ovalCenterY + offsetOvalCenterY) + config.maxOffsetY) {
      instruction += instruction ? " & up" : "Move up";
    }
  }
  
  if (isSecondCapture) {
    instructionText.innerText = instruction || "Position your face in the wider oval";
  } else {
    instructionText.innerText = instruction || "Position your face in the oval";
  }
}

function cleanup() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }
  
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(track => track.stop());
  }
}

window.addEventListener('beforeunload', cleanup);
initialize();