import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load sample data
data = pd.read_csv("sample_data.csv")

# Convert priority_level to numeric (Low=1, Medium=2, High=3)
data['priority_level'] = data['priority_level'].map({'Low':1, 'Medium':2, 'High':3})

# Features (input) and target (output)
X = data[['processing_time_minutes','priority_level']]
y = data['completed']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, predictions))
