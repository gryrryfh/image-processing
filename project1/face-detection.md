# 얼굴 추출 및 구현 팀프로젝트
## 소스코드

``` python

let video = document.getElementById('videoInput');
let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let gray = new cv.Mat();
let cap = new cv.VideoCapture(video);
let faces = new cv.RectVector();
let classifier = new cv.CascadeClassifier();
classifier.load('haarcascade_frontalface_default.xml');
const FPS = 30;
function processVideo() {
try { if (!streaming) {
src.delete();
dst.delete();
gray.delete();
faces.delete();
classifier.delete();
return; }
let begin = Date.now();
cap.read(src);
src.copyTo(dst);
cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
for (let i = 0; i < faces.size(); ++i) {
let face = faces.get(i);
let point1 = new cv.Point(face.x, face.y);
let point2 = new cv.Point(face.x + face.width, face.y + face.height);
cv.rectangle(dst, point1, point2, [0, 255, 0, 255], 5);}
cv.imshow('canvasOutput', dst);
let delay = 1000/FPS - (Date.now() - begin);
setTimeout(processVideo, delay);} 
catch (err) {
utils.printError(err); } };
// schedule the first one.
setTimeout(processVideo, 0);

```