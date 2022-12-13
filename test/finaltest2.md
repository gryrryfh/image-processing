## 기말고사

### 2번

#### 45점

### 실행결과
![2022-12-13 (4)](https://user-images.githubusercontent.com/50912987/207221822-0e2a9a52-3532-4b48-8f3e-9024df77cad6.png)

### p5js https://editor.p5js.org/gryrryfh/sketches/5MYKWeokh
### 소스코드

``` p5js

sketch.js

let mobilenet;
function modelReady() {
  mobilenet.predict(puffin, gotResults);
}

function gotResults(error, results) {
  if (error) {
    console.error(error);
  } else {
    console.log(results);
    let label = results[0].label;
    let prob = results[0].confidence;
    fill(255);
    textSize(64);
    text(label, 10, height - 100);
  }
}

function preload() {
  puffin = loadImage('testimage/test2.jpg');
}

function setup() {
  createCanvas(640, 480);
  background(0);
  image(puffin, 0, 0, width, height);
  mobilenet = ml5.imageClassifier("https://teachablemachine.withgoogle.com/models/T1t6ZWb_m/", modelReady);
}

html

<html>
  <head>
    <meta charset="UTF-8" />
    <title>Image classification using MobileNet and p5.js</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.1/p5.js"></script>
    <script src="https://unpkg.com/ml5@0.12.2/dist/ml5.min.js"></script>
  </head>

  <body>
    <h1>얼굴 인식하기</h1>
    <script src="sketch.js"></script>
  </body>
</html>


```

