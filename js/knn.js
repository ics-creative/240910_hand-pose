const webcamElement = document.getElementById("webcam"); // video要素
const downloadButton = document.getElementById("download-button"); // ［download］ボタン
const poseList = ["ピース", "指ハート", "ほっぺハート"]; // デフォルトのポーズ名を定義

// ウェブカメラを有効にする関数
async function enableCam() {
  const constraints = {
    audio: false,
    video: { width: 640, height: 480 },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints); // ウェブカメラのストリームを取得
    webcamElement.srcObject = stream; // ストリームをvideo要素に設定

    await new Promise((resolve) => {
      webcamElement.onloadedmetadata = () => {
        webcamElement.play(); // ストリームの再生を開始
        resolve();
      };
    });

    return await tf.data.webcam(webcamElement); // TensorFlow.jsのウェブカメラデータソースを返す
  } catch (error) {
    console.error("Error accessing webcam: ", error);
    alert(
      "カメラのアクセスに失敗しました。カメラのアクセス権限を確認してください。",
    );
  }
}

// イベントリスナーを追加する関数
function addEventListeners(classifier, detector) {
  // デフォルトのボタンのイベントリスナーを追加
  document
    .getElementById("class-0")
    .addEventListener("click", () => addExample(classifier, 0, detector));
  document
    .getElementById("class-1")
    .addEventListener("click", () => addExample(classifier, 1, detector));
  document
    .getElementById("class-2")
    .addEventListener("click", () => addExample(classifier, 2, detector));

  // 新しいポーズを追加するためのフォーム
  const newPoseForm = document.getElementById("new-pose-form");
  // 新しいポーズを追加するための［追加］ボタン
  const addNewPoseButton = document.getElementById("add-new-pose-button");
  // 新しく追加されたポーズ名
  const newPoseNameInput = document.getElementById("new-pose-name");

  // ［download］ボタンをクリックすると学習結果がダウンロードされる
  downloadButton.addEventListener("click", () => {
    downloadModel(classifier);
  });

  // ［追加］ボタンを押すとユーザーが新規追加したボタンが登録される
  addNewPoseButton.addEventListener("click", () => {
    const newPoseName = newPoseNameInput.value.trim();
    if (newPoseName) {
      addClassButtonToDOM(newPoseName, 3, detector, classifier);
      poseList.push(newPoseName);
      newPoseNameInput.value = "";
      newPoseForm.style.display = "none"; // 新しいポーズのボタンが追加されたらフォームは非表示にする
    }
  });
}

// 新しい項目のボタンをDOMに追加する関数
function addClassButtonToDOM(newPoseName, classId, detector, classifier) {
  const button = document.createElement("button");
  button.classList.add("button"); // ボタンにクラスを追加
  button.id = `class-${classId}`; // 新しいIDを設定
  button.innerText = newPoseName; // ボタンのテキストとして新しいポーズ名を設定

  // ボタンがクリックされたときに、新しいポーズをKNN分類器に追加するイベントリスナーを設定
  button.addEventListener("click", () =>
    addExample(classifier, classId, detector),
  );

  // ボタンを既存のボタンリストに追加
  const buttonsDiv = document.querySelector(".buttons");
  buttonsDiv.insertBefore(button, downloadButton); // ［download］ボタンの直前に新しいボタンを挿入
}

// KNN分類器を準備する関数
async function setupKNN() {
  const classifier = knnClassifier.create(); // TensorFlow.jsのKNN分類器を作成
  return classifier;
}

// 手を検出するためのモデルを初期化する関数
async function createHandDetector() {
  // handPoseDetection はライブラリの機能
  const model = handPoseDetection.SupportedModels.MediaPipeHands; // MediaPipeHandsモデルを使用
  const detectorConfig = {
    runtime: "mediapipe", // or "tfjs", ランタイムの選択
    solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands", // MediaPipeHandsのソリューションパス
    modelType: "full", // モデルタイプを設定
  };
  // 手の検出器を作成
  const detector = await handPoseDetection.createDetector(
    model,
    detectorConfig,
  );

  return detector;
}

// ウェブカメラの映像から手を検出する関数
async function estimateHands(detector) {
  const results = await detector.estimateHands(webcamElement, {
    flipHorizontal: false,
  });
  return results;
}

// 手のキーポイント（x, y, z）座標を取得する関数
async function getHandKeypoints(results) {
  if (results.length > 0) {
    return results[0].keypoints3D.map((point) => [point.x, point.y, point.z]);
  }
  return null;
}

// 手のポーズを予測する関数
async function estimatePose(classifier, results) {
  if (classifier.getNumClasses() > 0) {
    const keypoints = await getHandKeypoints(results); // 手のキーポイントを取得

    // デフォルトの予測結果は「なし」とする
    let predictionText = "prediction: なし\nprobability: 1";

    // 手のキーポイントが検出された場合のみ予測を更新
    if (keypoints) {
      // キーポイントをフラット化（1次元配列に変換）
      const flattened = keypoints.flat();

      // フラット化した配列をテンソルに変換し、2次元の形に変形
      const tensor = tf.tensor(flattened).reshape([1, flattened.length]);

      // KNN分類器を使ってポーズを予測
      const result = await classifier.predictClass(tensor);

      predictionText = `prediction: ${poseList[result.label]}\nprobability: ${Math.round(result.confidences[result.label] * 100) / 100}`;

      tensor.dispose();
    }

    // 予測結果を表示
    document.getElementById("console").innerText = predictionText;
  }

  // 次のフレームで再度処理を行う
  await tf.nextFrame();
}

// ポーズの学習を追加する関数
async function addExample(classifier, classId, detector) {
  const results = await estimateHands(detector); // 手の検出結果を取得
  const keypoints = await getHandKeypoints(results); // 手のキーポイントを取得

  if (keypoints) {
    // キーポイントをフラット化（1次元配列に変換）
    const flattened = keypoints.flat();

    // フラット化した配列をテンソルに変換し、2次元の形に変形
    const tensor = tf.tensor(flattened).reshape([1, flattened.length]);

    classifier.addExample(tensor, classId); // KNN分類器にポーズを追加
    tensor.dispose();
  }
}

// KNNモデルをダウンロードする関数
function downloadModel(classifier) {
  // モデルのデータセットを取得し、JSON文字列に変換
  const str = JSON.stringify(
    Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [
      label,
      Array.from(data.dataSync()),
      data.shape,
    ]),
  );
  const blob = new Blob([str], { type: "text/plain" }); // JSON文字列をBlobとして作成
  const url = URL.createObjectURL(blob); // BlobからURLを作成

  // ダウンロード用のリンクを作成
  const a = document.createElement("a");
  a.href = url;
  a.download = "knn-classifier-model.txt";

  // リンクをドキュメントに追加してクリックイベントを発火
  document.body.appendChild(a);
  a.click();

  document.body.removeChild(a); // リンクをドキュメントから削除
  URL.revokeObjectURL(url); // 作成したURLを解放
}

// 毎フレーム走らせる処理
async function render(detector, classifier) {
  // 手を検出する
  const results = await estimateHands(detector);
  // 手のポーズを予測する
  await estimatePose(classifier, results);

  window.requestAnimationFrame(() => render(detector, classifier));
}

// 初期化関数
async function init() {
  await enableCam(); // ウェブカメラの起動
  const detector = await createHandDetector(); // 手検出モデルの初期化
  const classifier = await setupKNN(); // KNNモデルの準備

  addEventListeners(classifier, detector); // イベントリスナーの設定
  render(detector, classifier); // 毎フレーム走らせる処理
}

// 初期化関数を呼び出す
init();
