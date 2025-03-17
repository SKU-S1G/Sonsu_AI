from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import modules.holistic_module as hm
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",  # 모든 origin 허용
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# 설정
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
seq_length = 10
fontpath = '/Users/cuj/Library/Fonts/Pretendard-Bold.otf'
font = ImageFont.truetype(fontpath, 40)

# 모델 및 MediaPipe 초기화
detector = hm.HolisticDetector(min_detection_confidence=0.3)
interpreter = tf.lite.Interpreter(model_path="models/multi_hand_gesture_classifier.tflite")
interpreter.allocate_tensors()

# 입력 및 출력 정보
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

seq = []
action_seq = []
last_action = None
current_question = None
game_result = None

# 비디오 스트리밍 처리
def process_frame():
    global seq, action_seq, last_action, game_result
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        # 카메라 좌우 반전 처리
        img = cv2.flip(img, 1)  # 1은 좌우 반전

        img = detector.findHolistic(img, draw=True)
        _, right_hand_lmList = detector.findLefthandLandmark(img)

        if right_hand_lmList is not None:
            # 손 랜드마크 처리
            joint = np.zeros((42, 2))
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]
            vector, angle_label = Vector_Normalization(joint)
            d = np.concatenate([vector.flatten(), angle_label.flatten()])

            seq.append(d)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            y_pred = interpreter.get_tensor(output_details[0]['index'])
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[0][i_pred]

            if conf > 0.9:
                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action
                    if last_action != this_action:
                        last_action = this_action

                        # 게임 결과 판단
                        if current_question is not None:
                            if this_action == current_question:
                                game_result = "정답입니다!"
                            else:
                                game_result = "틀렸습니다!"


        # 프레임 반환
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask 라우트
@app.route('/')
def index():
    return render_template('game.html')

# 게임 상태 가져오기
@app.route('/get_game_info', methods=['GET'])
def get_game_info():
    global last_action, game_result
    return jsonify({
        'last_action': last_action,
        'game_result': game_result
    })


@app.route('/video_feed')
def video_feed():
    # 비디오 스트리밍 응답 헤더에 CORS 설정 추가
    return Response(process_frame(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Access-Control-Allow-Origin': '*',  # 모든 origin 허용
                        'Access-Control-Allow-Methods': 'GET, POST',  # 허용되는 메서드 추가
                        'Access-Control-Allow-Headers': 'Content-Type, Authorization'  # 허용되는 헤더 추가
                    })

@app.route('/get_question', methods=['GET'])
def get_question():
    global current_question, game_result
    current_question = random.choice(actions)
    game_result = None
    return jsonify({"question": current_question})


if __name__ == '__main__':
    app.run(
        debug=True, 
        host='0.0.0.0',  # 모든 네트워크 인터페이스에서 접근 허용
        port=5001
    )
