import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np

emotion_model_path = './model/mini_XCEPTION_emotion.h5'
emotion_classifier =load_model(emotion_model_path, compile=False)
EMOTIONS = ["anger", "neutral", "sad", "smile", "surprise"]

def emotion_recognition(box, img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped_img = gray[box[1]:box[3], box[0]:box[2]]
    roi = cv2.resize(cropped_img, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # 모델을 이용하여 감정을 예측하고 label을 return 한다.
    prediction = emotion_classifier.predict(roi)[0]
    label = EMOTIONS[prediction.argmax()]
    return label

