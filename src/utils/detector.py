import cv2

faceProto = './model/opencv_face_detector.pbtxt'
faceModel = './model/opencv_face_detector_uint8.pb'
threshold = 0.45

faceNet = cv2.dnn.readNet(faceModel, faceProto)

def face_detect(img_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1, (300, 300), False, False)

    faceNet.setInput(blob)
    detections = faceNet.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            check = True
            # detection 결과에서 박스의 위치가 0보다 작으면 값을 변경한다.
            for j in range(3, 7):
                if detections[0, 0, i, j] < 0 and j < 5:
                    detections[0, 0, i, j] = 0
                elif detections[0, 0, i, j] < 0 and j == 5:
                    detections[0, 0, i, j] = width
                elif detections[0, 0, i, j] < 0 and j == 6:
                    detections[0, 0, i, j] = height

                #  detection 결과에서 박스의 위치가 사진의 크기를 벗어나면 중단시킨다.
                if int(detections[0, 0, i, j] * width) > width and (j == 3 or j == 5):
                    check = False
                    break
                elif int(detections[0, 0, i, j] * height) > height and (j == 4 or j == 6):
                    check = False
                    break
            if check:
                boxes.append([int(detections[0, 0, i, 3] * width),
                              int(detections[0, 0, i, 4] * height),
                              int(detections[0, 0, i, 5] * width),
                              int(detections[0, 0, i, 6] * height)])
    return boxes

