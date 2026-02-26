from preprocess import Preprocessing
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix


TEXT_COL = "content"
NUM_COLS = ["title_word_count", "title_text_ratio"]


def build_pipeline():
    text_features = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000))
    ])

    num_features = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("text", text_features, TEXT_COL),
        ("num", num_features, NUM_COLS)
    ], remainder="drop")

    svm = LinearSVC()
    calibrated_svm = CalibratedClassifierCV(svm)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_svm)
    ])

    return pipeline


def train_and_save(train_df: pd.DataFrame, test_df: pd.DataFrame, model_path="model.joblib"):
    X_train = train_df[[TEXT_COL] + NUM_COLS]
    y_train = train_df["label"]

    X_test = test_df[[TEXT_COL] + NUM_COLS]
    y_test = test_df["label"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, model_path)
    print(f"Model saved → {model_path}")


pp = Preprocessing()

data = pd.read_csv("news_cleaned.csv")
data = pp.preprocess_dataframe(data)

train_data, test_data = pp.time_split(data)
print(train_data.shape, test_data.shape)

train_and_save(train_data, test_data)