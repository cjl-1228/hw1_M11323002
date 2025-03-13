# activation_function_comparison_enhanced.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. 建立較困難的資料集（加入雜訊）
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_informative=10,
                           flip_y=0.1, class_sep=0.7, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 加入雜訊強化難度
X_train += np.random.normal(0, 0.1, X_train.shape)
X_test += np.random.normal(0, 0.1, X_test.shape)

# 標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. 定義模型訓練函數（加層數、加Dropout）
def build_and_train_model(activation_fn):
    model = Sequential()
    model.add(Dense(64, activation=activation_fn, input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation=activation_fn))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation=activation_fn))
    model.add(Dense(1, activation='sigmoid'))  # 二元分類
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_test, y_test))

    # 預測與評估
    y_pred_train = model.predict(X_train).flatten() > 0.5
    y_pred_test = model.predict(X_test).flatten() > 0.5
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    return train_acc, test_acc, history

# 3. 比較不同激活函數
activations = ['sigmoid', 'softplus', 'relu']
results = {}

for act in activations:
    print(f"Training model with activation function: {act}")
    train_acc, test_acc, history = build_and_train_model(act)
    results[act] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'history': history.history
    }

# 4. 顯示結果
for act in activations:
    print(f"\nActivation: {act}")
    print(f"Training Accuracy: {results[act]['train_acc']:.4f}")
    print(f"Testing Accuracy:  {results[act]['test_acc']:.4f}")

# 5. 繪圖
plt.figure(figsize=(12, 6))
for act in activations:
    plt.plot(results[act]['history']['val_accuracy'], label=f'{act} (val acc)')
plt.title('Validation Accuracy vs Epochs (Enhanced Comparison)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
