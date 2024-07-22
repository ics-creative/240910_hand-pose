import {
  enableCam,
  createHandDetector,
  flattenAndConvertToTensor,
} from "./utils.js";
const decoLoadedImage = {}; // 画像を格納するオブジェクト
const decoImageList = ["peace01", "peace02", "heart01", "heart02", "heart03"]; // 画像のリスト
const webcamElement = document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
const canvasWrapperElement = document.getElementById("canvas-wrapper");
const ctx = canvasElement.getContext("2d");

// Canvasの初期化関数
function initCanvas() {
  // Canvasの大きさをvideo要素に合わせる
  canvasElement.width = webcamElement.videoWidth;
  canvasElement.height = webcamElement.videoHeight;
  canvasWrapperElement.style.width = `${webcamElement.videoWidth}px`;
  canvasWrapperElement.style.height = `${webcamElement.videoHeight}px`;
}

// ウェブカメラの画像をCanvasに描画する関数
function drawWebCamToCanvas() {
  ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // キャンバスの水平反転を設定
  ctx.save(); // 現在の状態を保存
  ctx.scale(-1, 1); // 水平反転
  ctx.translate(-canvasElement.width, 0); // 座標を移動して反転を適用

  ctx.drawImage(
    webcamElement,
    0,
    0,
    webcamElement.videoWidth,
    webcamElement.videoHeight,
  );

  ctx.restore(); // 反転を元に戻す
}

// KNNモデルを読み込む非同期関数
async function loadKNNModel() {
  const response = await fetch("models/knn-classifier-model.txt");
  const txt = await response.text();
  const classifier = knnClassifier.create(); // TensorFlow.jsのKNN分類器を作成

  // テキストをJSONとして解析し、各ラベルに対応するデータと形状を取得
  // 取得したデータをテンソルに変換してKNN分類器に設定
  // https://github.com/tensorflow/tfjs/issues/633
  classifier.setClassifierDataset(
    Object.fromEntries(
      JSON.parse(txt).map(([label, data, shape]) => [
        label, // ラベル（クラス名）
        tf.tensor(data, shape), // データをテンソルに変換
      ]),
    ),
  );

  return classifier;
}

// ウェブカメラの映像から手を検出する関数
async function estimateHands(detector) {
  const hands = await detector.estimateHands(webcamElement, {
    flipHorizontal: false,
  });
  return hands;
}

// 手のキーポイントの3D座標（x, y, z）を取得する関数
function getHandKeypoints3D(hands) {
  return hands.map((hand) =>
    hand.keypoints3D.map((point) => [point.x, point.y, point.z]),
  );
}

// 手のポーズを予測する関数
async function estimatePose(classifier, allHandKeypoints3D) {
  if (classifier.getNumClasses() > 0 && allHandKeypoints3D.length > 0) {
    return await Promise.all(
      allHandKeypoints3D.map(async (keypoints3D) => {
        const tensor = flattenAndConvertToTensor(keypoints3D); // キーポイントの3D座標をフラット化しテンソルに変換

        // KNN分類器を使ってポーズを予測
        const hand = await classifier.predictClass(tensor);

        const classes = ["ピース", "指ハート", "ほっぺハート", "なし"]; // 各ポーズ名を取得
        const probabilities = hand.confidences; // 各ポーズの確率を取得

        tensor.dispose();
        return {
          knnResult: classes[hand.label],
          knnProbability: probabilities[hand.label],
        };
      }),
    );
  }

  return [{ knnResult: "なし", knnProbability: 0 }];
}

// Canvasに画像を描画する関数
function drawCanvas(hands, poses) {
  if (!hands || hands.length === 0) return;

  hands.forEach((hand, index) => {
    const { keypoints, handedness } = hand;
    const { knnResult, knnProbability } = poses[index];

    // 手のキーポイントの2D座標（x, y）を名前（keypoint.name）から取得する関数
    const getKeypoint = (name) =>
      keypoints.find((keypoint) => keypoint.name === name);

    const wrist = getKeypoint("wrist"); // 手首
    const thumbTip = getKeypoint("thumb_tip"); // 親指の先端
    const indexFingerTip = getKeypoint("index_finger_tip"); // 人差し指の先端
    const middleFingerMcp = getKeypoint("middle_finger_mcp"); // 中指の中手指節関節（付け根の関節）
    const middleFingerTip = getKeypoint("middle_finger_tip"); // 中指の先端
    const pinkyFingerMcp = getKeypoint("pinky_finger_mcp"); // 小指の中手指節関節（付け根の関節）

    // 位置の中間点を計算
    const indexMiddleMidPointX = (indexFingerTip.x + middleFingerTip.x) / 2;
    const thumbIndexMidPointX = (thumbTip.x + indexFingerTip.x) / 2;
    const wristMiddleMidPointY = (middleFingerMcp.y + wrist.y) / 2;

    // 「どのポーズであるか」と「そのポーズである確率が1であるか」と「右手か左手か」で、画像と画像の貼る位置を変える
    if (knnProbability !== 1) return;

    if (knnResult === "ピース") {
      drawDecoImage({
        image: {
          Right: decoLoadedImage.peace01,
          Left: decoLoadedImage.peace02,
        }[handedness],
        x: indexMiddleMidPointX,
        y: indexFingerTip.y - 30,
      });
    } else if (knnResult === "指ハート") {
      drawDecoImage({
        image: decoLoadedImage.heart03,
        x: thumbIndexMidPointX,
        y: indexFingerTip.y - 30,
      });
    } else if (knnResult === "ほっぺハート") {
      drawDecoImage({
        image: {
          Right: decoLoadedImage.heart02,
          Left: decoLoadedImage.heart01,
        }[handedness],
        x: pinkyFingerMcp.x + (handedness === "Left" ? -30 : 0),
        y: wristMiddleMidPointY - 50,
      });
    }
  });
}

// 画像を読み込む関数
function loadDecoImages() {
  decoImageList.forEach((name) => {
    const img = new Image();
    img.src = `images/${name}.png`;
    decoLoadedImage[name] = img;
  });
}

// 画像を描画する関数
function drawDecoImage({ image, x, y }) {
  const flippedX = canvasElement.width - x;
  const dx = flippedX - image.width / 2; // 画像の中心に合わせるための計算
  const dy = y - image.height / 2; // 画像の中心に合わせるための計算

  ctx.save(); // 現在のキャンバス状態を保存
  ctx.translate(dx + image.width / 2, dy + image.height / 2); // 画像の中心に移動

  ctx.drawImage(
    image,
    -image.width / 2,
    -image.height / 2,
    image.width,
    image.height,
  );
  ctx.restore(); // 回転前の状態に戻す
}

// 毎フレーム走らせる処理
async function render(detector, classifier) {
  // 手を検出する
  const hands = await estimateHands(detector);
  // 手のキーポイントの3D座標を取得する
  const allHandKeypoints3D = getHandKeypoints3D(hands);
  // 手のポーズを予測する
  const poses = await estimatePose(classifier, allHandKeypoints3D);

  drawWebCamToCanvas(); // canvasにvideoを描画する
  drawCanvas(hands, poses); // canvasにやりたいことを描画する

  window.requestAnimationFrame(() => render(detector, classifier));
}

// 初期化関数
async function init() {
  loadDecoImages(); // 画像を読み込む
  await enableCam(webcamElement); // ウェブカメラの起動
  const detector = await createHandDetector(); // 手検出モデルの初期化
  const classifier = await loadKNNModel(); // KNNモデルを読み込む

  initCanvas(); // Canvasの初期化
  render(detector, classifier); // 毎フレーム走らせる処理
}

// 初期化関数を呼び出す
init();
