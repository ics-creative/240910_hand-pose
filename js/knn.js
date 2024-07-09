const webcamElement = document.getElementById('webcam');
const downloadButton = document.getElementById('download');
let classifier;
let net;
let webcam;
let detector;

// ダウンロードボタンのイベントリスナーを追加する関数
function addEventListeners() {
  downloadButton.addEventListener('click', () => {
    downloadModel();
  });
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

// KNNモデルを読み込む非同期関数
async function loadKNNModel() {
  const response = await fetch('models/knn-classifier-model.txt');
  const txt = await response.text();

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
    video: {width: 640, height: 480}
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

// メインアプリケーションの関数
async function app() {
  // 新しい例を追加する関数
  const addExample = async classId => {
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

  // ボタンのイベントリスナーを追加
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));
  document.getElementById('class-d').addEventListener('click', () => addExample(3));

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
        const classes = ['ピース', '指ハート', 'ほっぺハート', 'なし'];
        predictionText = `prediction: ${classes[result.label]}\n\nprobability: ${Math.round(result.confidences[result.label] * 100) / 100}`;

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
  await setupKNN(); // KNNモデルのセットアップ
  await loadKNNModel(); // モデルを読み込む
  await enableCam(); // ウェブカメラの初期化
  await createHandDetector(); // モデルの読み込み
  addEventListeners(); // イベントリスナーを設定
  app(); // メインアプリケーションを実行
}

// 初期化関数を呼び出す
init();
