import os
import xml.etree.ElementTree as Et
import cv2
import matplotlib.pyplot as plt

dataset_path = "../dataset/test_data/"

IMAGE_FOLDER = "img/"
ANNOTATIONS_FOLDER = "annotations/"

ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
img_root, amg_dir, img_files = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

fig = plt.figure(figsize=(20, 20))
i = 0
num = 0
# annotation 파일을 불러와서 일치하는 img 파일을 찾아낸다.
for xml_file in ann_files:
    try:
        img_name = img_files[img_files.index(".".join([xml_file[:-4], "jpg"]))]
    except:
        img_name = img_files[img_files.index(".".join([xml_file[:-4], "JPG"]))]
    img_file = os.path.join(img_root, img_name)
    img = cv2.imread(img_file)

    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    objects = root.findall("object")

    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        # 사진에 박스를 그리고, label을 표시한다.
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), int(round(int(height)/150)), 8)
        cv2.putText(img, name, (xmin, ymax+20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 0), 3, 8)

    # 사진을 12장씩 나타낸다.
    if i < 12:
        ax = fig.add_subplot(3, 4, i+1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        i += 1
    else:
        i = 0
        plt.savefig('../result/img/fig_{}.png'.format(num))
        num += 1
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(3, 4, i + 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        i += 1
plt.savefig('../result/img/fig_{}.png'.format(num))