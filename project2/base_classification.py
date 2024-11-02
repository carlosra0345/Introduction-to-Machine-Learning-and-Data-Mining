from dataset import glass_identification, X, targets
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, random_state=0)

# Baseline model that predicts the largest class
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)

# Predict and evaluate
baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
print(f"Baseline Most Frequent Class Accuracy: {baseline_accuracy:.2f}")