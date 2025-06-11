# 인공지능응용 프로젝트 PuppySense


## 1. 저장소 클론
git clone https://github.com/wonee1/Puppysense_BE.git
cd Puppysense_BE

## 2. 가상환경 설정
python -m venv venv
source venv/bin/activate  # 윈도우: venv\Scripts\activate

## 3. 의존성 설치
pip install -r requirements.txt

## 4. 서버 실행
flask run --host=0.0.0.0 --port=5000
또는
python app.py


## 주요 기능
EfficientNetB4 기반 감정 분석 모델 연동
이미지 전처리 및 추론 처리
AWS S3 이미지 저장 연동
프론트엔드와 연동 가능한 RESTful API 제공

