import re
import pandas as pd
import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

class Preprocessing:

    def clean_text(self, text: str) -> str:
        text = text.lower()   # lowercasing
        text = re.sub(r"\d+", "", text)   # Remove Numbers
        text = text.translate(str.maketrans("", "", string.punctuation))   # Remove Punctuation
        tokens = text.split()   # Tokenization
        tokens = [w for w in tokens if w not in stop_words]   # Remove Stopwords
        tokens = [lemmatizer.lemmatize(w) for w in tokens]    # Limmatization
        return " ".join(tokens)


    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["content"] = df["title"] + " " + df["text"]
        df["content"] = df["content"].astype(str).apply(self.clean_text)
        return df


    def time_split(self, df: pd.DataFrame, split_ratio=0.8):
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # Sort
        df = df.sort_values('date').reset_index(drop=True)

        # Quantile-based split
        split_end = int(len(df) * split_ratio)

        train_df = df.iloc[:split_end]
        test_df = df.iloc[split_end:]

        return train_df, test_df