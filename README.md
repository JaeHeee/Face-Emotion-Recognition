# Face Emotion Recognition

face emotion recognition
<br><br>
얼굴 데이터를 이용하여, emotion recognition에 대한 학습을 진행합니다. test data에서 얼굴을 검출해내고, 학습한 모델을 이용하여 emotion recognition을 합니다.
<br>
<br>
##### data load
    python load.py
xml 파일과 일치한 img파일을 찾아내서 감정폴더별로 나눕니다.
<br>
<br>
##### make annotations
    python make_xml.py
test_data 의 img에서 얼굴을 검출하여 바운딩 박스의 위치를 찾아내고, 감정을 분류합니다.
<br>
<br>
##### make result
    python predict_drawBox.py
xml 파일과 일치한 img파일을 찾아낸 후, annotation 정보를 이용하여, img에 바운딩 박스를 그리고, 감정 label을 표시합니다.
결과물은 result 폴더에 저장합니다.