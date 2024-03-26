if (!window.ImageProcessors) {
  window.ImageProcessors = {};
}
window.ImageProcessors.CamShift = async function camShift() {
  let video = document.getElementById("videoInput");
  let cap = new cv.VideoCapture(video);
  let frameNumber = 0;
  let arrOfFrames = [];
  let arrWindowBoxes = [];
  tracking(cap, frameNumber, arrOfFrames, arrWindowBoxes, video);
};

function tracking(cap, frameNumber, arrOfFrames, arrWindowBoxes, video) {
  let frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  cap.read(frame);
  arrOfFrames.push(frame);
  //console.log(arrOfFrames.length);
  //console.log(arrWindowBoxes.length);
  if (frame.empty()) return;
  if (arrWindowBoxes.length != 0) {
    preCamShift(cap, frame, frameNumber, arrOfFrames, arrWindowBoxes, video);
  } else {
    if (frameNumber % 40 == 0) {
      const worker = new Worker("worker.js");
      //console.log('start again');
      start(arrWindowBoxes, frame, worker); 
    }
    setTimeout(() => {
      tracking(cap, frameNumber, arrOfFrames, arrWindowBoxes, video);
    }, 50);
  }
  frameNumber++;
}

function preCamShift(
  cap,
  frame,
  frameNumber,
  arrOfFrames,
  arrWindowBoxes,
  video
) {
  if (arrWindowBoxes.length == 0) {
    tracking(cap, frameNumber, arrOfFrames, arrWindowBoxes, video);
    return;
  }
  let objects = arrWindowBoxes.shift();
  for (let i = 0; i < objects.length; i++) {
    let object = objects[i];
    //console.log("object.trackWindow", object.trackWindow);
    let roi = frame.roi(object.trackWindow);
    let hsvRoi = new cv.Mat();

    cv.cvtColor(roi, hsvRoi, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsvRoi, hsvRoi, cv.COLOR_RGB2HSV);

    let mask = new cv.Mat();
    let lowScalar = new cv.Scalar(30, 30, 0);
    let highScalar = new cv.Scalar(180, 180, 180);

    let low = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), lowScalar);
    let high = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), highScalar);

    cv.inRange(hsvRoi, low, high, mask);

    let roiHist = new cv.Mat();
    let hsvRoiVec = new cv.MatVector();
    hsvRoiVec.push_back(hsvRoi);
    cv.calcHist(hsvRoiVec, [0], mask, roiHist, [180], [0, 180]);
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX);
    roi.delete();
    hsvRoi.delete();
    mask.delete();
    low.delete();
    high.delete();
    hsvRoiVec.delete();

    let hsv = new cv.Mat(video.height, video.width, cv.CV_8UC3);
    let hsvVec = new cv.MatVector();
    hsvVec.push_back(hsv);

    object.hsvVec = hsvVec;
    object.roiHist = roiHist;
    object.hsv = hsv;
    object.dst = new cv.Mat();
    //console.log("object", object);
  }
  let termCrit = new cv.TermCriteria(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
    10,
    1
  );
  const FPS = 100;
  let hsv = new cv.Mat(video.height, video.width, cv.CV_8UC3);
  let countTracking = 0;
  camShift(
    cap,
    frameNumber,
    arrOfFrames,
    arrWindowBoxes,
    objects,
    termCrit,
    FPS,
    countTracking,
    video,
    hsv
  );
}

function camShift(
  cap,
  frameNumber,
  arrOfFrames,
  arrWindowBoxes,
  objects,
  termCrit,
  FPS,
  countTracking,
  video,
  hsv
) {
  if (!streaming) {
    // clean and stop
    hsv.delete();
    canvasContext.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
    return;
  }
  countTracking = countTracking + 1;
  let frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  cap.read(frame);
  frameNumber++;
  arrOfFrames.push(frame);
  if (frame.empty()) return;
  let currentFrameMat = arrOfFrames.shift();
  if (countTracking == 40) {
    //console.log("In countTracking");
    preCamShift(
      cap,
      currentFrameMat,
      frameNumber,
      arrOfFrames,
      arrWindowBoxes,
      video
    );
  } else {
    if (frameNumber % 40 == 0) {
      const worker = new Worker("worker.js");
      //console.log("In frameNumber");
      start(arrWindowBoxes, frame, worker);
    }
    hsv = new cv.Mat(video.height, video.width, cv.CV_8UC3);
    objects.forEach((object) => {
      cv.cvtColor(currentFrameMat, object.hsv, cv.COLOR_RGBA2RGB);
      cv.cvtColor(object.hsv, object.hsv, cv.COLOR_RGB2HSV);
      cv.calcBackProject(
        object.hsvVec,
        [0],
        object.roiHist,
        object.dst,
        [0, 180],
        1
      );
      let [trackBox, newTrackWindow] = cv.CamShift(
        object.dst,
        object.trackWindow,
        termCrit
      );
      object.trackWindow = newTrackWindow;
      let pts = cv.rotatedRectPoints(trackBox);
      cv.line(currentFrameMat, pts[0], pts[1], [255, 0, 0, 255], 3);
      cv.line(currentFrameMat, pts[1], pts[2], [255, 0, 0, 255], 3);
      cv.line(currentFrameMat, pts[2], pts[3], [255, 0, 0, 255], 3);
      cv.line(currentFrameMat, pts[3], pts[0], [255, 0, 0, 255], 3);
    });
    cv.imshow("canvasOutput", currentFrameMat);
    setTimeout(function () {
      camShift(
        cap,
        frameNumber,
        arrOfFrames,
        arrWindowBoxes,
        objects,
        termCrit,
        FPS,
        countTracking,
        video,
        hsv
      );
    }, 100);
  }
}

async function start(arrWindowBoxes, frame, worker) {
  const imageData = matToCanvas(frame);
  worker.postMessage(imageData);
  worker.onmessage = (event) => {
    const boxes = event.data;
    let objects_ = [];
    for (i = 0; i < boxes.length; i++) {
      let box = convertToXYWH(boxes[i]);
      let object = { trackWindow: box };
      objects_.push(object);
    }
    arrWindowBoxes.push(objects_);
    //console.log("arrWindowBoxes.len",arrWindowBoxes.length);
  };
}
function convertToXYWH(bbox) {
  const x1 = bbox[0];
  const y1 = bbox[1];
  const x2 = bbox[2];
  const y2 = bbox[3];
  const width = Math.abs(x2 - x1);
  const height = Math.abs(y2 - y1);
  return new cv.Rect(x1, y1, width, height);
  // return new cv.Rect(40, 70, 200, 100);
}
function matToCanvas(frame) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = frame.cols;
  canvas.height = frame.rows;
  const imageData = ctx.createImageData(frame.cols, frame.rows);
  const data = new Uint8ClampedArray(frame.data);
  imageData.data.set(data);
  ctx.putImageData(imageData, 0, 0);
  const imgData = ctx.getImageData(0, 0, 640, 640);
  return imgData;
}
