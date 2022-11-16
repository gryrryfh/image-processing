## 안면인식 구현

### 얼굴을 학습시켜 하나의 사진에 있는 여러 사람의 얼굴을 인식해보기

### 20191128 이재경

### 코드
```javascript
const imageUpload = document.getElementById('imageUpload')

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

async function start() {
  const container = document.createElement('div')
  container.style.position = 'relative'
  document.body.append(container)
  const labeledFaceDescriptors = await loadLabeledImages()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  let image
  let canvas
  document.body.append('Loaded')
  imageUpload.addEventListener('change', async () => {
    if (image) image.remove()
    if (canvas) canvas.remove()
    image = await faceapi.bufferToImage(imageUpload.files[0])
    container.append(image)
    canvas = faceapi.createCanvasFromMedia(image)
    container.append(canvas)
    const displaySize = { width: image.width, height: image.height }
    faceapi.matchDimensions(canvas, displaySize)
    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
      drawBox.draw(canvas)
    })
  })
}

function loadLabeledImages() {
  const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        descriptions.push(detections.descriptor)
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}
```
### 구현방법
1. VSCODE, NODEjs, 폴더 안에 있는 내용들을 모두 다운 받는다.
2. VSCODE를 실행한 후 Extensions에서 Code Runner와 Live Server를 다운받은 후 index.html을 liveserver로 실행한다.
3. 알맞은 사진을 선택하면 작동한다.

### 구현영상

https://user-images.githubusercontent.com/50912987/202202456-15dcbcdc-c0c2-4782-81dc-8c350dc8c4e8.mp4

#### 소감
3시간 정도 걸린 것 같습니다. 영상을 여러 번 보면서 코드를 최대한 제 것으로 만드려고 했습니다. 안보고 척척 코드를 작성할 정도는 안되지만 이 내용들이 어떻게 작동하고 왜 이런지에 대해서는 확실히 이해를 한 것 같습니다. 솔직히 이 퀴즈가 나오기 전까지 어떻게 팀 프로젝트를 해결해나갈까 생각이 많았습니다. 팀 프로젝트에 대해 막연히 주제만 정하고 어떻게 할지 걱정이 했었는데 이런 퀴즈를 통해 조금이나마 나아갈 방향을 정하고 자신감이 생겼습니다.  이 퀴즈를 잘 해결한 것처럼 앞으로 남은 내용들을 잘 해결해서 좋은 결과를 얻고 싶습니다. 더 노력하겠습니다. 감사합니다.

#### 출처
https://github.com/WebDevSimplified/Face-Recognition-JavaScript
https://youtu.be/AZ4PdALMqx0
