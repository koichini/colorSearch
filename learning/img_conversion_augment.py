from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["red", "blue", "yellow", "green", "purple"]
num_classes = len(classes)
image_size = 50
test_data = 100

# 画像読み込み
X_train = []
X_test = []
Y_train = []
Y_test = []

for index, classlabel in enumerate(classes):
    photos_dir = "./img/" + classlabel
    files = glob.glob(photos_dir + './*.jpg')
    for i, file in enumerate(files):
        if i >= 400:
            break
        img = Image.open(file)
        img = img.convert('RGB')
        img = img.resize((image_size, image_size))
        data = np.asarray(img)

        if i < test_data:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 20, 5):
                # 回転
                img_r = img.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # 反転
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

xy = (X_train, X_test, y_train, y_test)
# ファイルの保存
np.save('./color_aug.npy', xy)
