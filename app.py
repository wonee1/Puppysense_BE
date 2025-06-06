from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)

# AWS S3 설정
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# 모델 경로 및 클래스
MODEL_PATH = "puppySenseModel_2025-06-04.h5"
class_names = ['angry', 'happy', 'relaxed', 'sad']

# 모델 로드 - 호환성 문제 해결을 위한 수정
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"기본 로드 방식 실패: {str(e)}")
    try:
        # 대체 로드 방식 시도
        model = tf.saved_model.load(MODEL_PATH)
    except Exception as e:
        print(f"대체 로드 방식도 실패: {str(e)}")
        raise e

print("모델 로딩 완료")

def upload_to_s3(file_obj, emotion):
    """S3에 파일 업로드하고 URL 반환"""
    try:
        # 파일명 생성 (감정_날짜_UUID.확장자)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_extension = file_obj.filename.split('.')[-1]
        new_filename = f"{emotion}_{timestamp}_{str(uuid.uuid4())[:8]}.{file_extension}"
        
        # S3에 업로드
        s3.upload_fileobj(
            file_obj,
            os.getenv('S3_BUCKET'),
            new_filename,
            ExtraArgs={'ACL': 'public-read'}
        )
        
        # S3 URL 생성
        url = f"https://{os.getenv('S3_BUCKET')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{new_filename}"
        return url
    except Exception as e:
        print(f"S3 업로드 에러: {str(e)}")
        return None

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
        
        # 각 감정에 대한 확률을 퍼센트로 변환 (정수로 반올림)
        emotion_percentages = {
            'neutral': round(float(prediction[0][2] * 100)),  # relaxed를 neutral로 사용
            'anger': round(float(prediction[0][0] * 100)),    # angry
            'happiness': round(float(prediction[0][1] * 100)), # happy
            'sadness': round(float(prediction[0][3] * 100))   # sad
        }

        # 가장 높은 확률의 감정 찾기
        pred_index = np.argmax(prediction)
        predicted_emotion = class_names[pred_index]

        # S3에 업로드
        image_file.seek(0)  # 파일 포인터를 처음으로 되돌림
        s3_url = upload_to_s3(image_file, predicted_emotion)

        return jsonify({
            "prediction": predicted_emotion,
            "emotions": emotion_percentages,
            "image_url": s3_url
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) # 포트 일단 5000번으로 호강니 

