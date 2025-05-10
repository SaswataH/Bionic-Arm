import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('hand_data.csv')

# Separate features and labels
X = data.drop('label', axis=1)
y = data['label']

# Encode string labels into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model and label encoder
joblib.dump(model, 'hand_gesture_svm_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
