const newClassForm = document.getElementById("new-pose-form");
const webcamElement = document.getElementById("webcam");
const customClasses = ["ピース", "指ハート", "ほっぺハート"]; // デフォルトのクラスを定義

// イベントリスナーを追加する関数
function addEventListeners(detector, classifier) {
  // デフォルトのボタンのイベントリスナーを追加
  document
    .getElementById("class-a")
    .addEventListener("click", () => addExample(classifier, 0, detector));
  document
    .getElementById("class-b")
    .addEventListener("click", () => addExample(classifier, 1, detector));
  document
    .getElementById("class-c")
    .addEventListener("click", () => addExample(classifier, 2, detector));

  // ダウンロードボタン
  const downloadButton = document.getElementById("download-button");

  // 新しいポーズを追加するためのボタン
  const addPoseButton = document.getElementById("add-button");
  const newPoseNameInput = document.getElementById("new-pose-name");

  // ダウンロードボタンをクリックすると学習結果がダウンロードされる
  downloadButton.addEventListener("click", () => {
    downloadModel();
  });

  // ［追加］ボタンを押すとユーザーが新規追加したボタンが登録される
  addPoseButton.addEventListener("click", () => {
    const className = newPoseNameInput.value.trim();
    if (className) {
      addClassButtonToDOM(className);
      customClasses.push(className);
      newPoseNameInput.value = "";
      newClassForm.style.display = "none";
    }
  });
}

// 新しい項目のボタンをDOMに追加する関数
function addClassButtonToDOM(className, downloadButton) {
  const button = document.createElement("button");
  button.classList.add("button");
  button.innerText = className;
  button.addEventListener("click", () =>
    addExample(customClasses.indexOf(className)),
  );

  const buttonsDiv = document.querySelector(".buttons");
  buttonsDiv.insertBefore(button, downloadButton);
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

// KNN分類器をセットアップする関数
async function setupKNN() {
  const classifier = knnClassifier.create(); // KNN分類器を作成
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
  detector = await handPoseDetection.createDetector(model, detectorConfig);

  return detector;
}

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

// ウェブカメラの映像から手のキーポイントを取得する関数
async function getHandKeypoints(detector) {
  const hands = await detector.estimateHands(webcamElement);
  if (hands.length > 0) {
    return hands[0].keypoints3D.map((point) => [point.x, point.y, point.z]);
  }
  return null;
}

// 新しいポーズの学習を追加する関数
async function addExample(classifier, classId, detector) {
  const landmarks = await getHandKeypoints(detector); // 手のキーポイントを取得

  if (landmarks) {
    // キーポイントをフラット化（1次元配列に変換）
    const flattened = landmarks.flat();

    // フラット化した配列をテンソルに変換し、2次元の形に変形
    const tensor = tf.tensor(flattened).reshape([1, flattened.length]);

    classifier.addExample(tensor, classId); // KNN分類器に新しいポーズを追加
    tensor.dispose();
  }
}

// 手のポーズを予測する関数
async function estimatePose(classifier, detector) {
  // 手のポーズを予測
  while (true) {
    if (classifier.getNumClasses() > 0) {
      const landmarks = await getHandKeypoints(detector, webcamElement); // 手のキーポイントを取得

      // デフォルトの予測結果は「なし」とする
      let predictionText = "prediction: なし\nprobability: 1";

      // 手のキーポイントが検出された場合のみ予測を更新
      if (landmarks) {
        // キーポイントをフラット化（1次元配列に変換）
        const flattened = landmarks.flat();

        // フラット化した配列をテンソルに変換し、2次元の形に変形
        const tensor = tf.tensor(flattened).reshape([1, flattened.length]);

        const result = await classifier.predictClass(tensor);
        predictionText = `prediction: ${customClasses[result.label]}\nprobability: ${Math.round(result.confidences[result.label] * 100) / 100}`;

        tensor.dispose();
      }

      // 予測結果を表示
      document.getElementById("console").innerText = predictionText;
    }

    await tf.nextFrame();
  }
}

// 初期化関数
async function init() {
  await enableCam();
  const detector = await createHandDetector();
  const classifier = await setupKNN();

  addEventListeners(detector, classifier);
  estimatePose(classifier, detector);
}

// 初期化関数を呼び出す
init();
