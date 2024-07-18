const decoLoadedImage = {}; // 画像を格納するオブジェクト
const decoImageList = ["peace01", "peace02", "heart01", "heart02", "heart03"]; // 画像のリスト
const webcamElement = document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
const canvasWrapperElement = document.getElementById("canvas-wrapper");
const ctx = canvasElement.getContext("2d");

// Webカメラを有効にする関数
async function enableCam() {
  const constraints = {
    audio: false,
    video: { width: 640, height: 480 },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    webcamElement.srcObject = stream;

    await new Promise((resolve) => {
      webcamElement.onloadedmetadata = () => {
        webcamElement.play();
        resolve();
      };
    });

    return await tf.data.webcam(webcamElement);
  } catch (error) {
    console.error("Error accessing webcam: ", error);
    alert(
      "カメラのアクセスに失敗しました。カメラのアクセス権限を確認してください。",
    );
  }
}

// Canvasの初期化関数
function initCanvas() {
  //canvasの大きさをwebcamに合わせる
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
  const classifier = knnClassifier.create(); // KNN分類器を作成

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

// 手を検知するためのモデルを初期化する関数
async function createHandDetector() {
  const model = handPoseDetection.SupportedModels.MediaPipeHands;
  const detectorConfig = {
    runtime: "mediapipe", // or 'tfjs',
    solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
    modelType: "full",
  };
  const detector = await handPoseDetection.createDetector(
    model,
    detectorConfig,
  );

  return detector;
}

// 画像から手のキーポイントを取得する関数
async function getHandKeypoints(imageElement, detector) {
  const hands = await detector.estimateHands(imageElement);
  if (hands.length > 0) {
    return hands[0].keypoints3D.map((point) => [point.x, point.y, point.z]);
  }
  return null;
}

// 手を検知する関数
async function estimateHands(detector) {
  const estimationConfig = { flipHorizontal: false };

  const results = await detector.estimateHands(webcamElement, estimationConfig);

  return results;
}

// 手のポーズを予測する関数
async function estimatePose(classifier, detector) {
  if (classifier.getNumClasses() > 0) {
    const keypoints = await getHandKeypoints(webcamElement, detector); // 手のキーポイントを取得

    if (keypoints) {
      // キーポイントをフラット化（1次元配列に変換）
      const flattened = keypoints.flat();

      // フラット化した配列をテンソルに変換し、2次元の形に変形
      const tensor = tf.tensor(flattened).reshape([1, flattened.length]);

      // KNN分類器を使ってポーズを予測
      const result = await classifier.predictClass(tensor);

      const classes = ["ピース", "指ハート", "ほっぺハート", "なし"]; // 各ポーズ名を取得
      const probabilities = result.confidences; // 各ポーズの確率を取得

      tensor.dispose();

      return {
        knnResult: classes[result.label],
        knnProbability: probabilities[result.label],
      };
    }

    return { knnResult: "なし", knnProbability: 0 }; // 何も検出されなかった場合のデフォルト値
  }

  // 次のフレームで再度処理を行う
  await tf.nextFrame();
}

// Canvasに画像を描画する関数
function drawCanvas(results, knnResult, knnProbability) {
  if (!results || results.length === 0) return;

  results.forEach((result) => {
    const { keypoints, handedness } = result;

    // 手のキーポイントを名前（keypoint.name）から取得
    const wrist = keypoints.find((keypoint) => keypoint.name === "wrist");
    const thumbTip = keypoints.find(
      (keypoint) => keypoint.name === "thumb_tip",
    );
    const indexFingerTip = keypoints.find(
      (keypoint) => keypoint.name === "index_finger_tip",
    );
    const middleFingerMcp = keypoints.find(
      (keypoint) => keypoint.name === "middle_finger_mcp",
    );
    const middleFingerTip = keypoints.find(
      (keypoint) => keypoint.name === "middle_finger_tip",
    );
    const pinkyFingerMcp = keypoints.find(
      (keypoint) => keypoint.name === "pinky_finger_mcp",
    );

    // 位置の中間点を計算
    const indexMiddleMidPointX = (indexFingerTip.x + middleFingerTip.x) / 2;
    const thumbIndexMidPointX = (thumbTip.x + indexFingerTip.x) / 2;
    const wristMiddleMidPointY = (middleFingerMcp.y + wrist.y) / 2;

    // 「どのポーズであるか」と「そのポーズである確率が1であるか」と「右手か左手か」で、画像と画像の貼る位置を変える
    if (
      knnResult === "ピース" &&
      knnProbability === 1 &&
      handedness === "Right"
    ) {
      drawDecoImage({
        image: decoLoadedImage.peace01,
        x: indexMiddleMidPointX,
        y: indexFingerTip.y - 30,
        scale: 3,
      });
    } else if (
      knnResult === "ピース" &&
      knnProbability === 1 &&
      handedness === "Left"
    ) {
      drawDecoImage({
        image: decoLoadedImage.peace02,
        x: indexMiddleMidPointX,
        y: indexFingerTip.y - 30,
        scale: 3,
      });
    } else if (knnResult === "指ハート" && knnProbability === 1) {
      drawDecoImage({
        image: decoLoadedImage.heart03,
        x: thumbIndexMidPointX,
        y: indexFingerTip.y - 30,
        scale: 2,
      });
    } else if (
      knnResult === "ほっぺハート" &&
      knnProbability === 1 &&
      handedness === "Right"
    ) {
      drawDecoImage({
        image: decoLoadedImage.heart02,
        x: pinkyFingerMcp.x,
        y: wristMiddleMidPointY - 50,
        scale: 2,
      });
    } else if (
      knnResult === "ほっぺハート" &&
      knnProbability === 1 &&
      handedness === "Left"
    ) {
      drawDecoImage({
        image: decoLoadedImage.heart01,
        x: pinkyFingerMcp.x - 30,
        y: wristMiddleMidPointY - 50,
        scale: 2,
      });
    }
  });
}

// 画像をロードする関数
function loadDecoImages() {
  decoImageList.forEach((name) => {
    const img = new Image();
    img.src = `images/${name}.png`;
    decoLoadedImage[name] = img;
  });
}

// 画像を描画する関数
function drawDecoImage({ image, x, y, scale = 1, xFix = 0, yFix = 0 }) {
  const flippedX = canvasElement.width - x;
  const dx = flippedX - image.width / scale / 2; // 画像の中心に合わせるための計算
  const dy = y - image.height / scale / 2; // 画像の中心に合わせるための計算

  ctx.save(); // 現在のキャンバス状態を保存
  ctx.translate(
    dx + xFix + image.width / scale / 2,
    dy + yFix + image.height / scale / 2,
  ); // 画像の中心に移動

  ctx.drawImage(
    image,
    -image.width / scale / 2,
    -image.height / scale / 2,
    image.width / scale,
    image.height / scale,
  );
  ctx.restore(); // 回転前の状態に戻す
}

// 毎フレーム走らせる処理
async function render(detector, classifier) {
  const { knnResult, knnProbability } = await estimatePose(
    classifier,
    detector,
  ); // ポーズを検知する
  const results = await estimateHands(detector); // 手を検知する
  drawWebCamToCanvas(); // canvasにvideoを描画する
  drawCanvas(results, knnResult, knnProbability); // canvasにやりたいことを描画する

  window.requestAnimationFrame(() => render(detector, classifier));
}

// 初期化関数
async function initHandPose() {
  loadDecoImages(); // 画像をロード
  await enableCam(); // Webカメラの起動
  const detector = await createHandDetector(); // 手検知モデルの初期化
  const classifier = await loadKNNModel(); // KNNモデルのロード

  initCanvas(); // Canvasの初期化
  render(detector, classifier); // 毎フレーム走らせる処理
}

// 初期化関数を呼び出す
initHandPose();
