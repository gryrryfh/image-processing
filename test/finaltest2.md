## 기말고사

### 2번

#### 45점

### 실행결과
#### 샘플들
![샘플](https://user-images.githubusercontent.com/50912987/207224026-54e468a0-0099-4ed3-9afe-8586672d08ff.png)
#### 결과
![결과](https://user-images.githubusercontent.com/50912987/207223705-5a910b3d-846b-4e66-aa9b-64b8753d0c1d.png)
![결과2](https://user-images.githubusercontent.com/50912987/207223918-44a300ee-13d6-4f99-8572-22b350dc40ca.PNG)

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

