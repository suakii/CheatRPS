import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import random

LABELS_EN = ["paper", "rock", "scissors"]
LABELS_KR = ["보", "바위", "가위"]

# 모델 로드
model = tf.keras.models.load_model("rps_mobilenetv2_1120.h5")

def get_computer_choice(user_choice_index):
    if user_choice_index == 2:
        return 1  # 바위
    elif user_choice_index == 1:
        return 0  # 보
    elif user_choice_index == 0:
        return 2  # 가위
    return random.randint(0, 2)

# 이미지 예측
def predict_image(image, model):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    user_choice_index = np.argmax(predictions)
    return predictions[0], user_choice_index

# Streamlit UI
st.title("반칙 가위바위보 게임")

# 카메라 입력
camera = st.camera_input("손 모양을 촬영하세요!")
if camera:
    file_bytes = np.asarray(bytearray(camera.getvalue()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
else:
    input_image = None

if camera and input_image is not None:
    probabilities, user_choice_index = predict_image(input_image, model)
    user_choice_kr = LABELS_KR[user_choice_index]  # 한글 라벨

    st.write(f"**사용자 선택:** {user_choice_kr} "
             f"(보 확률: {probabilities[0]:.2%}, 바위 확률: {probabilities[1]:.2%}, 가위 확률: {probabilities[2]:.2%})")

    col3, col_vs, col4 = st.columns([4, 1, 4])

    with col3:
        user_image_path = f"{LABELS_EN[user_choice_index]}.png"
        st.image(user_image_path, caption=f"사용자 선택: {LABELS_KR[user_choice_index]}", use_column_width=True)

    with col_vs:
        st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)

    with col4:
        computer_choice_index = get_computer_choice(user_choice_index)
        computer_choice_en = LABELS_EN[computer_choice_index]  # 영어
        computer_image_path = f"{computer_choice_en}.png"
        st.image(computer_image_path, caption=f"컴퓨터 선택: {LABELS_KR[computer_choice_index]}", use_column_width=True)

    # 결과 출력
    result = "무승부입니다!"
    if (user_choice_index == 2 and computer_choice_index == 1) or \
       (user_choice_index == 1 and computer_choice_index == 0) or \
       (user_choice_index == 0 and computer_choice_index == 2):
        result = "컴퓨터가 이겼습니다!"
    elif user_choice_index != computer_choice_index:
        result = "사용자가 이겼습니다!"
    st.header(f"**{result}**")
