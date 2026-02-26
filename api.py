from preprocess import Preprocessing
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, computed_field, Field
from typing import Annotated


app = FastAPI()
model = joblib.load("model.joblib")


class NewsInput(BaseModel):
    title: Annotated[str, Field(..., min_length=5, description="Title of the News")]
    text: Annotated[str, Field(..., min_length=50, description="Text of the News article.")]

    @computed_field
    @property
    def title_word_count(self) -> float:
        return np.log1p(len(self.title.split()))

    @computed_field
    @property
    def title_text_ratio(self) -> float:
        denom = np.log1p(len(self.text.split()))
        return self.title_word_count / max(denom, 1e-6)


@app.post("/predict")
def predict(news: NewsInput):
    content = news.title + " " + news.text
    pp = Preprocessing()
    content = pp.clean_text(str(content))

    X = [[
        content,
        news.title_word_count,
        news.title_text_ratio
    ]]

    X_df = pd.DataFrame(X, columns=["content", "title_word_count", "title_text_ratio"])

    pred = model.predict(X_df)[0]
    proba = model.predict_proba(X_df)[0]
    confidence = float(np.max(proba)*100)

    return {
        "prediction": str(pred),
        "confidence": confidence
    }