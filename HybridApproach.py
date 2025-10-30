import re
import math
import random
from collections import defaultdict
import pandas as pd
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK data not found. Please run the required NLTK downloads (see dependencies).")
    exit()

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return lemmatized_tokens

class NaiveBayesClassifier:
    def __init__(self, smoothing=1):
        self.smoothing = smoothing
        self.vocab = set()
        self.log_prior = {}
        self.log_likelihood = {'fake': defaultdict(float), 'real': defaultdict(float)}
        self.log_likelihood_unknown = {'fake': 0.0, 'real': 0.0}
        self.class_counts = {'fake': 0, 'real': 0}

    def train(self, documents, labels):
        word_counts = {'fake': defaultdict(int), 'real': defaultdict(int)}
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = preprocess_text(doc)
            for word in words:
                self.vocab.add(word)
                word_counts[label][word] += 1
        
        total_docs = len(documents)
        if total_docs == 0:
            print("Warning: NaiveBayesClassifier trained on 0 documents.")
            return

        self.log_prior['fake'] = math.log(self.class_counts['fake'] / total_docs)
        self.log_prior['real'] = math.log(self.class_counts['real'] / total_docs)
        
        vocab_size = len(self.vocab)
        total_words_fake = sum(word_counts['fake'].values())
        total_words_real = sum(word_counts['real'].values())

        denom_fake = total_words_fake + self.smoothing * vocab_size
        denom_real = total_words_real + self.smoothing * vocab_size
        
        self.log_likelihood_unknown['fake'] = math.log(self.smoothing / denom_fake)
        self.log_likelihood_unknown['real'] = math.log(self.smoothing / denom_real)

        for word in self.vocab:
            self.log_likelihood['fake'][word] = math.log((word_counts['fake'][word] + self.smoothing) / denom_fake)
            self.log_likelihood['real'][word] = math.log((word_counts['real'][word] + self.smoothing) / denom_real)

    def predict(self, document):
        words = preprocess_text(document)
        log_posterior_fake = self.log_prior.get('fake', -1e100)
        log_posterior_real = self.log_prior.get('real', -1e100)

        for word in words:
            log_posterior_fake += self.log_likelihood['fake'].get(word, self.log_likelihood_unknown['fake'])
            log_posterior_real += self.log_likelihood['real'].get(word, self.log_likelihood_unknown['real'])

        return "fake" if log_posterior_fake > log_posterior_real else "real"

def evaluate_and_print_results(model_name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='fake', zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label='fake', zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label='fake', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=['real', 'fake'])

    print(f"\n--- Evaluation Results for: {model_name} ---")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (for 'fake' class)")
    print(f"  Recall:    {recall:.4f} (for 'fake' class)")
    print(f"  F1-Score:  {f1:.4f} (for 'fake' class)")
    print(f"  Confusion Matrix:\n     \t  (Pred Real) (Pred Fake)")
    print(f"   (True Real) {cm[0][0]:>5}       {cm[0][1]:>5}")
    print(f"   (True Fake) {cm[1][0]:>5}       {cm[1][1]:>5}")
    print("-" * 45)

def main():
    DATA_FILE_REAL = "True.csv"
    DATA_FILE_FAKE = "Fake.csv"

    TEXT_COLUMN_NAME = 'text'

    MODEL_PARAMS = {
        'test_split_size': 0.25,
        'random_state_seed': 42
    }

    print("Starting the Fake News Detection Project Pipeline...")

    try:
        df_real = pd.read_csv(DATA_FILE_REAL)
        df_fake = pd.read_csv(DATA_FILE_FAKE)
    except FileNotFoundError:
        print(f"\n--- !!! ERROR !!! ---")
        print(f"Data file not found. Check these paths:")
        print(f"  - {DATA_FILE_REAL}")
        print(f"  - {DATA_FILE_FAKE}")
        print("Please update the file paths in the configuration block.")
        print("-" * 45)
        return
    except Exception as e:
        print(f"\n--- !!! ERROR !!! ---")
        print(f"An error occurred while loading the data: {e}")
        print("-" * 45)
        return

    df_real['label'] = 'real'
    df_fake['label'] = 'fake'
    df = pd.concat([df_real, df_fake], ignore_index=True)

    try:
        df = df.rename(columns={TEXT_COLUMN_NAME: 'text'})
    except KeyError:
        print(f"\n--- !!! ERROR !!! ---")
        print(f"Text column mapping failed. A column named '{TEXT_COLUMN_NAME}' was not found.")
        print("Please check your TEXT_COLUMN_NAME configuration.")
        print("-" * 45)
        return

    df.dropna(subset=['text', 'label'], inplace=True)
    if df.empty:
        print("\n--- !!! ERROR !!! ---\nNo valid data found after loading.")
        return
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], 
            df['label'], 
            test_size=MODEL_PARAMS['test_split_size'], 
            random_state=MODEL_PARAMS['random_state_seed'], 
            stratify=df['label']
        )
    except ValueError as e:
        print(f"\n--- !!! WARNING !!! ---")
        print(f"Could not stratify data split: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], 
            df['label'], 
            test_size=MODEL_PARAMS['test_split_size'], 
            random_state=MODEL_PARAMS['random_state_seed']
        )
        
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    print(f"\nData loaded and split: {len(df_train)} training samples, {len(df_test)} testing samples.")

    print("\n--- Starting Experiment 1: From-Scratch Naive Bayes ---")
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(df_train['text'], df_train['label'])
    predictions_nb = [nb_classifier.predict(doc) for doc in df_test['text']]
    evaluate_and_print_results("From-Scratch Naive Bayes", df_test['label'], predictions_nb)

    print("\n--- Starting Experiments 2 & 3: TF-IDF Benchmarks ---")
    print("Vectorizing text with TF-IDF (this may take a moment)...")
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text,
        lowercase=False,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=3
    )
    try:
        X_train_vec = vectorizer.fit_transform(df_train['text'])
        X_test_vec = vectorizer.transform(df_test['text'])
        print("Vectorizing complete.")
    except Exception as e:
        print(f"\n--- !!! ERROR !!! ---")
        print(f"TF-IDF vectorization failed: {e}")
        print("Skipping scikit-learn models.")
        print("-" * 45)
        X_train_vec = None

    if X_train_vec is not None:
        sklearn_nb = MultinomialNB(alpha=0.1)
        sklearn_nb.fit(X_train_vec, df_train['label'])
        predictions_sklearn_nb = sklearn_nb.predict(X_test_vec)
        evaluate_and_print_results("Scikit-learn MultinomialNB (TF-IDF)", df_test['label'], predictions_sklearn_nb)
        
        log_reg = LogisticRegression(
            random_state=MODEL_PARAMS['random_state_seed'], 
            max_iter=1000
        )
        log_reg.fit(X_train_vec, df_train['label'])
        predictions_log_reg = log_reg.predict(X_test_vec)
        evaluate_and_print_results("Scikit-learn Logistic Regression (TF-IDF)", df_test['label'], predictions_log_reg)

if __name__ == '__main__':
    main()