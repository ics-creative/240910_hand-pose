const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
const downloadButton = document.getElementById('download');
let net;
let knnResult;

function addEventListeners() {
  downloadButton.addEventListener('click', () => {
    downloadModel();
  });
}

// KNNモデルをダウンロードする関数
function downloadModel() {
  const str = JSON.stringify(
    Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [
      label,
      Array.from(data.dataSync()),
      data.shape,
    ])
  );
  const blob = new Blob([str], {type: 'text/plain'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'knn-classifier-model.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// KNNモデルを読み込む非同期関数
async function loadKNNModel() {
  const response = await fetch('models/knn-classifier-model.txt');
  const txt = await response.text();

  // https://github.com/tensorflow/tfjs/issues/633
  classifier.setClassifierDataset(
    Object.fromEntries(
      JSON.parse(txt).map(([label, data, shape]) => [
        label,
        tf.tensor(data, shape)
      ])
    )
  );

  return new Promise((resolve) => {
    resolve();
  });
}

async function app() {
  net = await mobilenet.load();

  const webcam = await tf.data.webcam(webcamElement);

  const addExample = async classId => {
    const img = await webcam.capture();

    const activation = net.infer(img, true);

    classifier.addExample(activation, classId);

    img.dispose();
  };

  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));
  document.getElementById('class-d').addEventListener('click', () => addExample(3));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      const activation = net.infer(img, 'conv_preds');
      const result = await classifier.predictClass(activation);

      const classes = ['ピース', '指ハート', 'ほっぺハート', 'なし'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${Math.round(result.confidences[result.label] * 100) / 100}
      `;

      knnResult = classes[result.label];

      img.dispose();
    }

    await tf.nextFrame();
  }
}

// 初期化関数
async function init() {
  await loadKNNModel() // モデルを読み込む
  addEventListeners(); // イベントリスナーを設定
  app();
}

// 初期化関数を呼び出す
init();
