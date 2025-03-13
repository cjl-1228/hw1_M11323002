# 匯入必要的函式庫
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

#載入 mnist 數據集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#資料前處理
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#定義函式來建立模型
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(10, activation='softmax', input_shape=(28 * 28,)))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#設定不同的 batch size & epoch
batch_sizes = [16, 32, 64]
epochs_list = [30, 60, 90]
results_dict = {}

for batch_size in batch_sizes:
    for epochs in epochs_list:
        print(f"\n🚀 訓練模型：batch_size={batch_size}, epochs={epochs}")
        
        model = build_model()  # 每次訓練都建立新的模型

        # 記錄訓練時間
        start_time = time.time()

        # 訓練模型
        history = model.fit(train_images, train_labels, 
                            epochs=epochs, batch_size=batch_size, 
                            validation_data=(test_images, test_labels),
                            verbose=0)
        
        # 計算測試集表現
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

        end_time = time.time()
        training_time = end_time - start_time  # 計算訓練時間

        # 儲存結果
        results_dict[(batch_size, epochs)] = {
            "Training Time": training_time,
            "Test Loss": test_loss,
            "Test Accuracy": test_acc
        }

        print(f"⏳ 訓練時間: {training_time:.2f} 秒, 🎯 測試準確率: {test_acc:.4f}, ❌ 測試損失: {test_loss:.4f}")

#視覺化分析不同 batch size & epoch 的影響
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#訓練時間
for batch_size in batch_sizes:
    times = [results_dict[(batch_size, epochs)]['Training Time'] for epochs in epochs_list]
    axes[0].plot(epochs_list, times, marker='o', label=f'Batch Size {batch_size}')

axes[0].set_title("Training Time vs Epochs")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Training Time (sec)")
axes[0].legend()
axes[0].grid()

#測試準確率 (Accuracy)
for batch_size in batch_sizes:
    acc_values = [results_dict[(batch_size, epochs)]['Test Accuracy'] for epochs in epochs_list]
    axes[1].plot(epochs_list, acc_values, marker='o', label=f'Batch Size {batch_size}')

axes[1].set_title("Test Accuracy vs Epochs")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid()

#測試損失 (Loss)
for batch_size in batch_sizes:
    loss_values = [results_dict[(batch_size, epochs)]['Test Loss'] for epochs in epochs_list]
    axes[2].plot(epochs_list, loss_values, marker='o', label=f'Batch Size {batch_size}')

axes[2].set_title("Test Loss vs Epochs")
axes[2].set_xlabel("Epochs")
axes[2].set_ylabel("Loss")
axes[2].legend()
axes[2].grid()

plt.show()