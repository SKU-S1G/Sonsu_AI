import cv2
import mediapipe as mp
import numpy as np
import time, os
from PIL import ImageFont, ImageDraw, Image

# 한글 텍스트 출력 함수
def draw_korean_text(img, text, position, font_path="malgun.ttf", font_size=40, color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

actions = [
    '안녕하세요', '감사합니다', '사랑합니다', '어머니', '아버지', '동생', '잘', '못', '간다', '나',
    '이름', '만나다', '반갑다', '부탁', '학교', '생일', '월', '일', '나이', '고발', '복습', '학습', '눈치', '오다', '말', '곱다',
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
]
seq_length = 30
secs_for_action = 30

# MediaPipe 모델 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()
        img = cv2.flip(img, 1)

        # 1. 대기 화면 (동작 이름 표시 + 카운트다운)
        for countdown in range(5, 0, -1):
            ret, img = cap.read()
            img = cv2.flip(img, 1)

            img = draw_korean_text(img, f'{action}', (int(img.shape[1]/2)-100, int(img.shape[0]/2)-50),
                                   font_size=50, color=(0, 255, 0))
            img = draw_korean_text(img, f'{countdown}초 후 시작', (int(img.shape[1]/2)-120, int(img.shape[0]/2)+20),
                                   font_size=30, color=(0, 255, 255))

            cv2.imshow('img', img)
            cv2.waitKey(1000)

        # 데이터 수집 안내 메시지
        img = draw_korean_text(img, f'{action} 동작 수집 중...', (10, 30), font_size=30, color=(255, 255, 255))
        cv2.imshow('img', img)
        cv2.waitKey(1000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = holistic.process(img_rgb)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            joint_list = []

            # 왼손
            if result.left_hand_landmarks:
                for lm in result.left_hand_landmarks.landmark:
                    joint_list.append([lm.x, lm.y, lm.z])
            else:
                joint_list.extend([[0, 0, 0]] * 21)

            # 오른손
            if result.right_hand_landmarks:
                for lm in result.right_hand_landmarks.landmark:
                    joint_list.append([lm.x, lm.y, lm.z])
            else:
                joint_list.extend([[0, 0, 0]] * 21)

            # 포즈
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    joint_list.append([lm.x, lm.y, lm.z])
            else:
                joint_list.extend([[0, 0, 0]] * 33)

            if joint_list:
                joint_list = np.array(joint_list).flatten()
                joint_list = np.append(joint_list, idx)
                data.append(joint_list)

            mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}'), data)

        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}'), full_seq_data)

    break

cap.release()
cv2.destroyAllWindows()
