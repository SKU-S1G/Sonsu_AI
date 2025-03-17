from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import random
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

# 모델 및 데이터 정보
model_path = 'final.tflite'  # TFLite 모델 경로
actions = [
    '안녕하세요', '감사합니다', '사랑합니다', '어머니', '아버지', '동생', '잘', '못', '간다', '나',
    '이름', '만나다', '반갑다', '부탁', '학교', '생일', '월', '일', '나이', '고발', '복습', '학습', '눈치채다', '오다', '말', '곱다'
]  # 학습한 동작 리스트
seq_length = 30  # 모델 학습 시 사용한 시퀀스 길이

# 한글 폰트 설정
font_path = "malgun.ttf"
font = ImageFont.truetype(font_path, 30)

current_question = None
game_result = None

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe 모델 로드
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 카메라 및 시퀀스 저장 리스트
cap = None
seq = []
is_recognizing = False

def draw_text(img, text, position, font, color=(0, 255, 0)):
    """PIL을 이용해 한글을 출력하는 함수"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

def generate_frames():
    """카메라 프레임을 스트리밍하는 함수"""
    global cap, seq, is_recognizing, current_question, game_result
    
    cap = cv2.VideoCapture(0)
    seq = []
    is_recognizing = True

    while is_recognizing:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = holistic.process(img_rgb)

        joint_list = []

        # 왼손 랜드마크 저장
        if result.left_hand_landmarks:
            for lm in result.left_hand_landmarks.landmark:
                joint_list.append([lm.x, lm.y, lm.z])
        else:
            joint_list.extend([[0, 0, 0]] * 21)

        # 오른손 랜드마크 저장
        if result.right_hand_landmarks:
            for lm in result.right_hand_landmarks.landmark:
                joint_list.append([lm.x, lm.y, lm.z])
        else:
            joint_list.extend([[0, 0, 0]] * 21)

        # 몸(상체) 랜드마크 저장
        if result.pose_landmarks:
            for lm in result.pose_landmarks.landmark:
                joint_list.append([lm.x, lm.y, lm.z])
        else:
            joint_list.extend([[0, 0, 0]] * 33)

        if joint_list:
            joint_list = np.array(joint_list).flatten()
            seq.append(joint_list)

            # 시퀀스 길이 유지
            if len(seq) > seq_length:
                seq.pop(0)

            # 예측 수행 (TFLite 모델 사용)
            if len(seq) == seq_length:
                input_data = np.expand_dims(np.array(seq), axis=0).astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0]

                predicted_action = actions[np.argmax(prediction)]
                confidence = np.max(prediction)

                color = (0, 255, 0)

                # 정답 여부 판별
                if current_question:
                    if predicted_action == current_question:
                        game_result = "정답입니다!"
                        color = (0, 255, 0)
                    else:
                        game_result = "틀렸습니다!"
                        color = (255, 0, 0)
                else:
                    game_result = "문제가 출제되지 않았습니다."

                # 한글 예측 결과 및 게임 결과 출력
                #img = draw_text(img, f'예측: {predicted_action} ({confidence:.2f})', (10, 50), font)
                #img = draw_text(img, f'문제: {current_question}', (10, 100), font)
                #img = draw_text(img, game_result, (10, 150), font, color)

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # 프레임을 웹에 전달
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('game1.html')

@app.route('/get_question', methods=['GET'])
def get_question():
    global current_question, game_result
    current_question = random.choice(actions)
    game_result = None
    return jsonify({"question": current_question})

@app.route('/start_recognition', methods=['GET'])
def start_recognition():
    global is_recognizing
    if not is_recognizing:
        thread = threading.Thread(target=generate_frames)
        thread.start()
    return jsonify({"status": "Recognition started"})

@app.route('/get_game_info', methods=['GET'])
def get_game_info():
    """게임 문제 및 결과를 반환하는 API"""
    global current_question, game_result
    return jsonify({"question": current_question, "game_result": game_result})

@app.route('/video_feed')
def video_feed():
    """웹캠 영상 스트리밍"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
