import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import CustomLinearRegression

# Dataset
df = pd.read_csv(fr"Dataset\admission.csv")
features = np.array(df.iloc[:, 1:-2])
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
label = np.array(df.iloc[:, -1])

# Training
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Initialize weights and bias
np.random.seed(0)
weights = np.random.rand(X_train.shape[1])
bias = np.random.rand()


# Usage
learning_rate=0.05
num_epochs=50
model = CustomLinearRegression(learning_rate=learning_rate, num_epochs=num_epochs)
model.fit(X_train, y_train, X_test, y_test)
y_pred = model.predict(X_test)

# Plotting Loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), model.train_losses, label='Training Loss', marker='o')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), model.test_losses, label='Testing Loss', marker='o')
plt.title('Testing Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

#GREScore  TOEFLScore  UniversityRating  SOP  LOR   CGP
print("hey:",model.predict(np.array([200, 95, 3, 4.5, 4.5, 5.5])))