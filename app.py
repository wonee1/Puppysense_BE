from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# 모델 경로 및 클래스
MODEL_PATH = "puppySenseModel_2025-06-01.keras"
class_names = ['angry', 'happy', 'relaxed', 'sad']

# 모델 로드
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ 모델 로딩 완료")

# 예측 API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 파일 가져오기
        image_file = request.files["image"]
        image = Image.open(BytesIO(image_file.read())).convert("RGB")

        # EfficientNetB4 입력에 맞게 리사이즈
        image = image.resize((224, 224))

        # 정규화는 모델 내부에서 처리 → float32로만 변환
        img_array = np.array(image).astype(np.float32)
        img_tensor = tf.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # 예측
        prediction = model.predict(img_tensor)
        pred_index = np.argmax(prediction)
        confidence = float(prediction[0][pred_index])

        return jsonify({
            "prediction": class_names[pred_index],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) # 포트 일단 5000번으로 호강니 

