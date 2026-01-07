import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import mlflow
import mlflow.sklearn

mlflow.set_experiment("DT_Model")

df = pd.read_csv("college_student_placement_dataset.csv")
df.drop(columns=["College_ID"], inplace=True)

X = df.drop(columns=["Placement"])
y = df["Placement"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


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

n_estimators = 500
max_depth = 7

with mlflow.start_run(run_name='Student_Placement_DT'):
    model = DecisionTreeClassifier(
        max_depth=max_depth
    )

    model.fit(X_train_enc, y_train_trans)
    y_pred = model.predict(X_test_enc)
    accuracy = accuracy_score(y_test_trans, y_pred)

    # mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
