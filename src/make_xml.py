import os
import cv2
from xml.etree.ElementTree import Element, SubElement, ElementTree
from utils import detector, predict

dataset_path = "../dataset/test_data/"

IMAGE_FOLDER = "img/"
ANNOTATIONS_FOLDER = "annotations/"

img_root, amg_dir, img_files = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

# 테스트 이미지 파일들을 불러온다.
for img_file in img_files:
    filename = img_file
    img_path = dataset_path + IMAGE_FOLDER + filename
    img = cv2.imread(img_path)

    width = img.shape[1]
    height = img.shape[0]
    # face detection을 해서 box의 위치를 찾아낸다.
    boxes = detector.face_detect(img_path)

    # annotation 파일을 생성한다.
    root = Element('annotation')
    SubElement(root, 'folder').text = 'test_img'
    SubElement(root, 'filename').text = filename
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    for box in boxes:
        obj = SubElement(root, 'object')
        label = predict.emotion_recognition(box, img_path)
        SubElement(obj, 'name').text = label
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(box[0])
        SubElement(bbox, 'ymin').text = str(box[1])
        SubElement(bbox, 'xmax').text = str(box[2])
        SubElement(bbox, 'ymax').text = str(box[3])

    tree = ElementTree(root)
    tree.write(dataset_path + ANNOTATIONS_FOLDER + filename[:-4] + '.xml')
