from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np


classes = ["red", "blue", "yellow", "green", "purple"]
num_classes = len(classes)
image_size = 50


def main():
    X_train, X_test, y_train, y_test = np.load("./color.npy", allow_pickle=True)
    # データの正規化
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    # on-hot-vector
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # モデルの学習
    model = model_train(X_train, y_train)
    # モデルの評価
    model_eval(model, X_test, y_test)


def model_train(X, y):
    '''
    https://github.com/valohai/keras-example/blob/master/cifar10_cnn.py
    padding='same' -> 畳み込み結果が同じサイズになるようにぷくセルを足す
    input_shape -> 入力データ（画像）の形状
    X.shape[] -> (450, 50, 50, 3) -> (450個の 50*50*3の配列)
    X.shaoe[1:] -> (50, 50, 3)
    '''
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    # 最適化
    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    # モデルの呼び出し
    model.fit(X, y, batch_size=32, epochs=100)

    # モデルの保存
    model.save('./color_cnn.h5')

    return model


def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('test loss: ', scores[0])
    print('test accuracy: ', scores[1])


# Pythonから直接ファイルを呼ばれた時の処理
if __name__ == "__main__":
    main()