from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd

# 載入 MNIST 資料
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 強化模型差異（加層數、調 optimizer 與 epochs）
def build_and_train_model(activation_function):
    model = models.Sequential()
    model.add(layers.Dense(128, activation=activation_function, input_shape=(784,)))
    model.add(layers.Dense(128, activation=activation_function))
    model.add(layers.Dense(64, activation=activation_function))
    model.add(layers.Dense(64, activation=activation_function))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_images, train_labels, epochs=20, batch_size=256, validation_split=0.2, verbose=0)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    return history, test_acc

# 比較不同激活函數
activation_functions = ["sigmoid", "softplus", "relu"]
results = {}

for func in activation_functions:
    history, test_acc = build_and_train_model(func)
    results[func] = {"history": history, "test_accuracy": test_acc}

# 繪圖（Validation Accuracy）
plt.figure(figsize=(8, 6))
for func in activation_functions:
    plt.plot(results[func]["history"].history["val_accuracy"], label=f"{func}")
plt.title("Validation Accuracy Comparison (MNIST)")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 顯示測試集準確率
for func in activation_functions:
    print(f"{func} activation - Test Accuracy: {results[func]['test_accuracy']:.4f}")
