import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import mlflow
import mlflow.sklearn
import dagshub

# DAGsHub + MLflow Setup
dagshub.init(
    repo_owner='Rohitpatil1304',
    repo_name='MLops_Experiment-Tracking_dagshub',
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Rohitpatil1304/MLops_Experiment-Tracking_dagshub.mlflow"
)

mlflow.set_experiment("DT_Model")

# Data Loading
df = pd.read_csv("college_student_placement_dataset.csv")
df.drop(columns=["College_ID"], inplace=True)

X = df.drop(columns=["Placement"])
y = df["Placement"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Encoding
le = LabelEncoder()
y_train_trans = le.fit_transform(y_train)
y_test_trans = le.transform(y_test)

ohe = OneHotEncoder(
    sparse_output=False,
    drop="first",
    handle_unknown="ignore"
)

X_train_enc = ohe.fit_transform(X_train)
X_test_enc = ohe.transform(X_test)

# Model Parameters 
max_depth = 7

# MLflow Run 
with mlflow.start_run(run_name="Student_Placement_DT"):

    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train_enc, y_train_trans)

    y_pred = model.predict(X_test_enc)
    accuracy = accuracy_score(y_test_trans, y_pred)

    # Metrics & Params 
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # Confusion Matrix 
    cm = confusion_matrix(y_test_trans, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Decision Tree")

    # Save artifact
    artifact_path = "confusion_matrix.png"
    plt.savefig(artifact_path)
    plt.close()

    # Log artifact to MLflow
    mlflow.log_artifact(artifact_path)

    # Log model
    mlflow.sklearn.log_model(model, "decision_tree_model")

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
