from dataset import glass_identification, X, targets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# One-hot encode targets for multiclass classification
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(targets.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

# Define and train the ANN model
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200)
mlp_model.fit(X_train, y_train)

# Make predictions
y_pred = mlp_model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_test_classes = y_test.argmax(axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"ANN Model Accuracy: {accuracy:.2f}")

