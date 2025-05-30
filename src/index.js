import express from "express";
import multer from "multer";
import fs from "fs";
import * as tf from "@tensorflow/tfjs";
import jpeg from "jpeg-js";
import path from "path";
import cors from "cors";
import { fileURLToPath } from "url";

// __dirname 대체
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
const upload = multer({ dest: "uploads/" });


// 모델과 클래스 이름
let model;
const classNames = ["happy", "sad", "angry", "neutral"]; // 학습 클래스에 맞게 수정

// 모델 로딩
async function loadModel() {
  const modelPath = `file://${__dirname}/tfjs_model/model.json`;
  model = await tf.loadLayersModel(modelPath);
  console.log("✅ 모델 로드 완료");
}

// 이미지 전처리 함수
function preprocessImage(imagePath) {
  const buf = fs.readFileSync(imagePath);
  const pixels = jpeg.decode(buf, true);

  const tensor = tf.tensor3d(pixels.data, [pixels.height, pixels.width, 3])
    .resizeNearestNeighbor([112, 112])
    .toFloat()
    .div(tf.scalar(255))
    .expandDims(); // [1, 112, 112, 3]

  return tensor;
}

// 예측 API
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    const tensor = preprocessImage(req.file.path);
    const prediction = model.predict(tensor);
    const predArray = await prediction.data();

    const maxIndex = predArray.indexOf(Math.max(...predArray));

    res.json({
      prediction: classNames[maxIndex],
      confidence: predArray[maxIndex],
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "예측 실패" });
  } finally {
    fs.unlinkSync(req.file.path); // 이미지 삭제
  }
});

// 정적 파일 (index.html 등) 서비스
app.use(express.static(path.join(__dirname, "public")));

// 서버 시작
loadModel().then(() => {
  app.listen(3000, () => {
    console.log("🚀 서버 실행 중: http://localhost:3000");
  });
});
