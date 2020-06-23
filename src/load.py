import os
import xml.etree.ElementTree as Et
from PIL import Image

dataset_path = "../dataset/train/"

IMAGE_FOLDER = "img"
ANNOTATIONS_FOLDER = "annotations"
train_path = "../dataset/train_data"

ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
img_root, amg_dir, img_files = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

emotions = ['neutral', 'anger', 'surprise', 'smile', 'sad']
# emotion 폴더 생성
for em in emotions:
  if not os.path.exists('{}/{}'.format(train_path, em)):
    os.mkdir('{}/{}'.format(train_path, em))

print("start")
num = 0
# annotation 파일을 불러와서 일치하는 img 파일을 찾아낸다.
for xml_file in ann_files:
    try:
      img_name = img_files[img_files.index(".".join([xml_file[:-4], "jpg"]))]
    except:
      img_name = img_files[img_files.index(".".join([xml_file[:-4], "JPG"]))]
    img_file = os.path.join(img_root, img_name)

    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    objects = root.findall("object")

    for _object in objects:
        emotion = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # box의 위치를 찾아내서 img를 자르고 감정별로 폴더에 저장한다.
        img = Image.open(img_file)
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        resized_img = cropped_img.resize((64, 64))
        name = "{}.jpg".format("{0:04d}".format(num))
        path = os.path.join(train_path, emotion, name)
        resized_img.save(path)
        num += 1

print("end")