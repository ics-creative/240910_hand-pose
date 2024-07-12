const webcamElement = document.getElementById('webcam');
const downloadButton = document.getElementById('download');
const addClassButton = document.getElementById('add-class');
const newClassNameInput = document.getElementById('new-class-name');
const newClassForm = document.getElementById('new-class-form');
let classifier;
let net;
let webcam;
let detector;
let customClasses = ['ピース', '指ハート', 'ほっぺハート']; // デフォルトのクラスを定義

// ダウンロードボタンのイベントリスナーを追加する関数
function addEventListeners() {
  downloadButton.addEventListener('click', () => {
    downloadModel();
  });

  addClassButton.addEventListener('click', () => {
    const className = newClassNameInput.value.trim();
    if (className) {
      addClassButtonToDOM(className);
      customClasses.push(className);
      newClassNameInput.value = '';
      newClassForm.style.display = 'none';
    }
  });
}

// 新しいポーズのボタンをDOMに追加する関数
function addClassButtonToDOM(className) {
  const button = document.createElement('button');
  button.classList.add('button');
  button.innerText = className;
  button.addEventListener('click', () => addExample(customClasses.indexOf(className)));

  // downloadボタンの前に新しいボタンを追加
  const buttonsDiv = document.querySelector('.buttons');
  buttonsDiv.insertBefore(button, downloadButton);
}

// KNNモデルをダウンロードする関数
function downloadModel() {
  // モデルのデータセットを取得し、JSON文字列に変換
  const str = JSON.stringify(
    Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [
      label,
      Array.from(data.dataSync()),
      data.shape,
    ])
  );
  const blob = new Blob([str], {type: 'text/plain'}); // JSON文字列をBlobとして作成
  const url = URL.createObjectURL(blob); // BlobからURLを作成

  // ダウンロード用のリンクを作成
  const a = document.createElement('a');
  a.href = url;
  a.download = 'knn-classifier-model.txt';

  // リンクをドキュメントに追加してクリックイベントを発火
  document.body.appendChild(a);
  a.click();

  document.body.removeChild(a); // リンクをドキュメントから削除
  URL.revokeObjectURL(url); // 作成したURLを解放
}

// KNN分類器とMobileNetモデルをセットアップする関数
async function setupKNN() {
  classifier = knnClassifier.create(); // KNN分類器を作成
  net = await mobilenet.load(); // MobileNetモデルをロード

  return new Promise((resolve) => {
    resolve();
  });
}

// 手を検知するためのモデルを初期化する関数
async function createHandDetector() {
  const model = handPoseDetection.SupportedModels.MediaPipeHands;
  const detectorConfig = {
    runtime: 'mediapipe', // or 'tfjs',
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands',
    modelType: 'full',
  }
  detector = await handPoseDetection.createDetector(model, detectorConfig);

  return new Promise((resolve) => {
    resolve(detector);
  });
}

// Webカメラを有効にする関数
async function enableCam() {
  const constraints = {
    audio: false,
    video: {width: 640, height: 480},
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    webcamElement.srcObject = stream;

    return new Promise((resolve) => {
      webcamElement.onloadedmetadata = async () => {
        webcamElement.play();
        webcam = await tf.data.webcam(webcamElement); // ウェブカメラの初期化
        resolve();
      };
    });
  } catch (error) {
    console.error('Error accessing webcam: ', error);
    alert('カメラのアクセスに失敗しました。カメラのアクセス権限を確認してください。');
  }
}

// 画像から手のランドマークを取得する関数
async function getHandLandmarks(imageElement) {
  const hands = await detector.estimateHands(imageElement);
  if (hands.length > 0) {
    return hands[0].keypoints3D.map(point => [point.x, point.y, point.z]);
  }
  return null;
}

// 新しい例を追加する関数
async function addExample(classId) {
  const img = await webcam.capture();
  const landmarks = await getHandLandmarks(webcamElement);

  if (landmarks) {
    const flattened = landmarks.flat();
    const tensor = tf.tensor(flattened).reshape([1, flattened.length]);

    classifier.addExample(tensor, classId);
    tensor.dispose();
  }

  img.dispose();
};

// メインアプリケーションの関数
async function app() {
  // デフォルトのボタンのイベントリスナーを追加
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  // 手のポーズを予測
  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();
      const landmarks = await getHandLandmarks(webcamElement);

      // デフォルトの予測結果は「なし」とする
      let predictionText = 'prediction: なし\n\nprobability: 1';

      // 手のランドマークが検出された場合のみ予測を更新
      if (landmarks) {
        const flattened = landmarks.flat();
        const tensor = tf.tensor(flattened).reshape([1, flattened.length]);

        const result = await classifier.predictClass(tensor);
        predictionText = `prediction: ${customClasses[result.label]}\n\nprobability: ${Math.round(result.confidences[result.label] * 100) / 100}`;

        tensor.dispose();
      }

      // 予測結果を表示
      document.getElementById('console').innerText = predictionText;
      img.dispose();
    }

    await tf.nextFrame();
  }
}

// 初期化関数
async function init() {
  // Webカメラの起動、手検知モデルの初期化、KNNモデルのセットアップを並列に実行
  await Promise.all([
    enableCam(),           // Webカメラの起動
    createHandDetector(),  // 手検知モデルの初期化
    setupKNN()             // KNNモデルのセットアップ
  ]);

  addEventListeners(); // イベントリスナーを設定
  app(); // メインアプリケーションを実行
}

// 初期化関数を呼び出す
init();
