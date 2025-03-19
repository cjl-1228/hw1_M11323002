import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=seed)
X_reg = StandardScaler().fit_transform(X_reg)
y_reg = (y_reg - np.mean(y_reg)) / np.std(y_reg)  


train_data, test_data, train_targets, test_targets = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=seed
)


regression_results = {'Activation': [], 'MAE': [], 'RMSE': [], 'MAPE': []}


def build_regression_model(activation):
    model = models.Sequential()
    model.add(layers.Dense(64, activation=activation, input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation=activation))
    model.add(layers.Dense(1)) 
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


for act in ['sigmoid', 'softplus', 'relu']:
    model = build_regression_model(act)
    model.fit(train_data, train_targets, epochs=50, batch_size=16, verbose=0)
    predictions = model.predict(test_data).flatten()  
    
    mae = mean_absolute_error(test_targets, predictions)
    rmse = np.sqrt(mean_squared_error(test_targets, predictions))
    mape = mean_absolute_percentage_error(test_targets, predictions)
    
    regression_results['Activation'].append(act)
    regression_results['MAE'].append(mae)
    regression_results['RMSE'].append(rmse)
    regression_results['MAPE'].append(mape)


fig, axs = plt.subplots(1, 3, figsize=(18, 5))


axs[0].plot(regression_results['Activation'], regression_results['MAE'], marker='o', linestyle='-')
axs[0].set_title('Regression Model - MAE')
axs[0].set_ylabel('Mean Absolute Error (MAE)')
axs[0].set_xlabel('Activation Function')

axs[1].plot(regression_results['Activation'], regression_results['RMSE'], marker='o', linestyle='-')
axs[1].set_title('Regression Model - RMSE')
axs[1].set_ylabel('Root Mean Squared Error (RMSE)')
axs[1].set_xlabel('Activation Function')

axs[2].plot(regression_results['Activation'], regression_results['MAPE'], marker='o', linestyle='-')
axs[2].set_title('Regression Model - MAPE')
axs[2].set_ylabel('Mean Absolute Percentage Error (MAPE)')
axs[2].set_xlabel('Activation Function')

plt.tight_layout()
plt.show()
