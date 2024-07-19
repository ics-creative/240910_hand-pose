/** ウェブカメラを有効にする関数 **/
export async function enableCam(webcamElement) {
  const constraints = {
    audio: false,
    video: { width: 640, height: 480 },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    webcamElement.srcObject = stream;

    await new Promise((resolve) => {
      webcamElement.onloadedmetadata = () => {
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

/** 手を検出するためのモデルを初期化する関数 **/
export async function createHandDetector() {
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
