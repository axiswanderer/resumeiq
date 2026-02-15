import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.features import process_resumes

# 1. Define what we are looking for
REQUIRED_SKILLS = ["Python", "Machine Learning", "SQL", "Docker"]
RESUME_DIR = "data/raw/resumes"

print("Parsing resumes and extracting features...")
X_features = process_resumes(RESUME_DIR, REQUIRED_SKILLS)

# 2. Load the Ground Truth (Y)
y_labels = pd.read_csv("data/processed_data.csv")

# 3. Merge Features with Labels
# We match them on 'resume_id' to ensure data integrity
full_data = pd.merge(X_features, y_labels[['resume_id', 'hired_label']], on='resume_id')

# 4. Prepare for Training
# Drop non-numerical columns
drop_cols = ['resume_id', 'text', 'hired_label']
X = full_data.drop(columns=drop_cols)
y = full_data['hired_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train XGBoost
print("Training XGBoost...")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 6. Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"✅ Model Trained! Accuracy: {acc * 100:.2f}%")
print("Feature Importances:")
print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))

# 7. Save the model for the app to use
model.save_model("resume_model.json")
print("💾 Model saved to resume_model.json")