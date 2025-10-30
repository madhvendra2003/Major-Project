"""
================================================================================
Project Documentation: A Hybrid Approach to Fake News Detection
================================================================================

AUTHOR: [Your Name Here]
DATE: October 30, 2025
VERSION: 1.7 (Removed GloVe, Cleaned Pipeline)

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------
This project implements and compares a comprehensive suite of "content-only" 
models for automated fake news detection. It serves as a complete experimental
pipeline for a research paper, comparing models of increasing complexity.

The models included are:
1.  MODULE 1A: From-Scratch Naive Bayes
    A frequency-based baseline, made numerically stable with log-probabilities.

2.  MODULE 1B: Custom Word2Vec Analyzer
    A novel model that first trains its own 'Word2Vec' embeddings from the
    dataset, then uses those custom embeddings for classification.

3.  MODULE 3 (SKLEARN): Optimized ML Benchmarks
    Industry-standard TF-IDF vectorization paired with Multinomial Naive Bayes
    and Logistic Regression (our primary benchmark).

4.  MODULE 3 (KERAS): Deep Learning Model
    A research-grade Bidirectional LSTM (Long Short-Term Memory) network
    to capture sequential context and nuanced meaning.

The main() function loads a two-file dataset (real/fake), trains all four
model types, and provides a clear, comparative performance report.
"""

# --- DEPENDENCIES ---
# pip install pandas scikit-learn nltk tensorflow
#
# After installation, run python and type:
# import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

import re
import math
import random
from collections import defaultdict
import pandas as pd
import numpy as np  # Required for Keras

# NLTK
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Embedding, LSTM, Bidirectional, Dense, Dropout,
        Input, Flatten, Dot
    )
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    print("--- TENSORFLOW NOT FOUND ---")
    print("This script requires TensorFlow to run the LSTM and Word2Vec models.")
    print("Please install it: pip install tensorflow")
    exit()

# --- Global Initializations for Efficiency ---
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK data not found. Please run the required NLTK downloads (see dependencies).")
    exit()

# --- MODULE 0: DATA PREPROCESSING ---

def preprocess_text(text):
    """
    Cleans and tokenizes text data.
    - Converts to lowercase
    - Removes non-alphabetic characters
    - Tokenizes
    - Removes stopwords
    - Lemmatizes
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return lemmatized_tokens

# --- MODULE 1A: Baseline Content Analyst (From-Scratch Naive Bayes) ---
class NaiveBayesClassifier:
    """A from-scratch Naive Bayes classifier using log-probabilities."""
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

# --- MODULE 1B: Custom Word2Vec Vector Analyst ---
class CustomWord2VecAnalyzer:
    """
    A model that first trains its own Word2Vec embeddings from the dataset,
    then uses them for classification via prototype averaging.
    """
    def __init__(self, embedding_dim=50, window_size=5, min_count=2, epochs=3):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.epochs = epochs
        
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        
        self.word_embeddings = {} # This is what we will train
        self.model_weights = {'fake': [], 'real': []} # Prototypes
        self.zero_vector = [0.0] * self.embedding_dim

    def _build_vocab(self, documents):
        print("Building Word2Vec vocabulary...")
        all_tokens = []
        word_counts = defaultdict(int)
        
        for doc in documents:
            tokens = preprocess_text(doc)
            all_tokens.append(tokens)
            for token in tokens:
                word_counts[token] += 1
        
        vocab_index = 0
        self.word_to_id['<UNK>'] = vocab_index # Unknown word token
        self.id_to_word[vocab_index] = '<UNK>'
        vocab_index += 1
        
        for word, count in word_counts.items():
            if count >= self.min_count:
                self.word_to_id[word] = vocab_index
                self.id_to_word[vocab_index] = word
                vocab_index += 1
                
        self.vocab_size = len(self.word_to_id)
        print(f"Vocabulary built. Size: {self.vocab_size} unique words.")
        return all_tokens

    def _generate_skipgram_pairs(self, tokenized_documents):
        print("Generating Skip-gram training pairs...")
        skip_gram_pairs = []
        for doc_tokens in tokenized_documents:
            for i, center_word_token in enumerate(doc_tokens):
                if center_word_token not in self.word_to_id:
                    center_word_token = '<UNK>'
                
                center_word_id = self.word_to_id[center_word_token]
                
                for j in range(max(0, i - self.window_size), min(len(doc_tokens), i + self.window_size + 1)):
                    if i == j: continue
                    context_word_token = doc_tokens[j]
                    if context_word_token not in self.word_to_id:
                        context_word_token = '<UNK>'
                    
                    context_word_id = self.word_to_id[context_word_token]
                    skip_gram_pairs.append([center_word_id, context_word_id])
                    
        return np.array(skip_gram_pairs)

    def _build_and_train_word2vec(self, training_data):
        print("Building Keras Word2Vec model...")
        
        input_center = Input((1,))
        input_context = Input((1,))
        
        embedding_layer = Embedding(self.vocab_size, 
                                    self.embedding_dim, 
                                    name="word2vec_embedding")
        
        center_embedding = embedding_layer(input_center)
        center_embedding = Flatten()(center_embedding)
        
        context_embedding = embedding_layer(input_context)
        context_embedding = Flatten()(context_embedding)
        
        dot_product = Dot(axes=1)([center_embedding, context_embedding])
        output = Dense(1, activation='sigmoid')(dot_product)
        
        model = Model(inputs=[input_center, input_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        
        center_words = training_data[:, 0]
        context_words = training_data[:, 1]
        
        # Generate negative samples
        num_negative_samples = len(training_data)
        fake_center = np.random.randint(0, self.vocab_size, num_negative_samples)
        fake_context = np.random.randint(0, self.vocab_size, num_negative_samples)
        
        x_center = np.concatenate([center_words, fake_center])
        x_context = np.concatenate([context_words, fake_context])
        y_labels = np.concatenate([np.ones(len(training_data)), np.zeros(num_negative_samples)])
        
        print(f"Training Word2Vec model on {len(x_center)} pairs...")
        model.fit([x_center, x_context], y_labels,
                  epochs=self.epochs,
                  batch_size=128,
                  verbose=1)
        
        # Extract the trained weights
        trained_weights = embedding_layer.get_weights()[0]
        for word, word_id in self.word_to_id.items():
            self.word_embeddings[word] = trained_weights[word_id]
        
        print("Custom Word2Vec embeddings trained and extracted.")

    def _get_doc_vector(self, document):
        tokens = preprocess_text(document)
        if not tokens:
            return self.zero_vector
        
        doc_vector_sum = [0.0] * self.embedding_dim
        words_found = 0
        
        for token in tokens:
            vector = self.word_embeddings.get(token, self.zero_vector)
            doc_vector_sum = [sum(x) for x in zip(doc_vector_sum, vector)]
            if token in self.word_embeddings:
                words_found += 1
        
        if words_found > 0:
            return [x / words_found for x in doc_vector_sum]
        else:
            return self.zero_vector

    def train(self, documents, labels):
        # Part A: Train the Word2Vec Embeddings
        tokenized_documents = self._build_vocab(documents)
        training_data = self._generate_skipgram_pairs(tokenized_documents)
        if len(training_data) == 0:
            print("No training data generated for Word2Vec. Aborting.")
            return
        self._build_and_train_word2vec(training_data)
        
        # Part B: Train the Averaging Classifier
        print("Training the averaging classifier using custom embeddings...")
        class_vectors = defaultdict(list)
        for doc, label in zip(documents, labels):
            doc_vector = self._get_doc_vector(doc)
            if any(v != 0.0 for v in doc_vector):
                class_vectors[label].append(doc_vector)

        for label, vectors in class_vectors.items():
            if vectors:
                self.model_weights[label] = [sum(col) / len(col) for col in zip(*vectors)]
            else:
                self.model_weights[label] = self.zero_vector
        print("Classifier training complete.")

    def _cosine_similarity(self, vec1, vec2):
        if not isinstance(vec1, list) or not isinstance(vec2, list): return 0
        if len(vec1) != self.embedding_dim or len(vec2) != self.embedding_dim: return 0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def predict(self, document):
        doc_vector = self._get_doc_vector(document)
        sim_fake = self._cosine_similarity(doc_vector, self.model_weights.get('fake', self.zero_vector))
        sim_real = self._cosine_similarity(doc_vector, self.model_weights.get('real', self.zero_vector))
        return "fake" if sim_fake > sim_real else "real"


# --- MODULE 3: EVALUATION & KERAS LSTM MODEL ---

def evaluate_and_print_results(model_name, y_test, y_pred):
    """Prints a formatted report of classification metrics."""
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

def build_and_train_lstm(X_train, y_train, X_test, y_test):
    """
    Builds, trains, and evaluates a Bidirectional LSTM model.
    """
    
    # --- 1. Define Model Parameters ---
    VOCAB_SIZE = 10000     # Max number of words to keep
    MAX_SEQ_LENGTH = 250   # Max number of words per article (padding)
    EMBEDDING_DIM = 128    # Dimension of word vectors
    
    # --- 2. Preprocessing for Keras ---
    print("Preprocessing text for Keras LSTM...")
    # Keras Tokenizer learns the vocabulary
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences of integers
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to be the same length
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    
    # Convert labels from 'real'/'fake' to 0/1
    y_train_binary = np.array([1 if label == 'fake' else 0 for label in y_train])
    y_test_binary = np.array([1 if label == 'fake' else 0 for label in y_test])

    # --- 3. Build the Model ---
    print("Building LSTM model...")
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, 
                  output_dim=EMBEDDING_DIM, 
                  input_length=MAX_SEQ_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # --- 4. Train the Model ---
    print("Training LSTM model (this will take several minutes)...")
    history = model.fit(
        X_train_pad, 
        y_train_binary,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # --- 5. Evaluate the Model ---
    print("Evaluating LSTM model...")
    y_pred_probs = model.predict(X_test_pad)
    y_pred_lstm = ["fake" if prob > 0.5 else "real" for prob in y_pred_probs]
    
    # Use our existing evaluation function
    evaluate_and_print_results("Keras Bidirectional LSTM", y_test, y_pred_lstm)
    

# --- MODULE 4: MAIN EXECUTION PIPELINE ---

def main():
    """Main function to run the experimental pipeline."""
    
    # ----------------------------------------------------------------------
    # --- USER CONFIGURATION ---
    # Edit these variables to match your dataset.
    # ----------------------------------------------------------------------

    # 1. Specify the path to your TWO dataset files
    DATA_FILE_REAL = "True.csv"
    DATA_FILE_FAKE = "Fake.csv"

    # 2. Map your dataset's text column name
    TEXT_COLUMN_NAME = 'text'

    # 3. Set model parameters
    MODEL_PARAMS = {
        'test_split_size': 0.25, # Use 25% of data for testing
        'random_state_seed': 42  # For reproducible results
    }

    # ----------------------------------------------------------------------
    # --- END CONFIGURATION ---
    # Do not edit below this line
    # ----------------------------------------------------------------------
    
    print("Starting the Fake News Detection Project Pipeline...")
    print("NOTE: Running in CONTENT-ONLY mode.")

    # 1. Load and Prepare Data
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
    
    # 2. Split Data
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

    # --- 3. Run Experiments (From-Scratch Models) ---

    # Experiment 1: From-Scratch Naive Bayes (Frequency Baseline)
    print("\n--- Starting Experiment 1: From-Scratch Naive Bayes ---")
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(df_train['text'], df_train['label'])
    predictions_nb = [nb_classifier.predict(doc) for doc in df_test['text']]
    evaluate_and_print_results("From-Scratch Naive Bayes", df_test['label'], predictions_nb)

    # Experiment 2: Custom Word2Vec Analyzer (Semantic Baseline)
    print("\n--- Starting Experiment 2: Custom Word2Vec Analyzer ---")
    try:
        w2v_analyzer = CustomWord2VecAnalyzer(embedding_dim=50, epochs=3)
        w2v_analyzer.train(df_train['text'], df_train['label'])
        predictions_w2v = [w2v_analyzer.predict(doc) for doc in df_test['text']]
        evaluate_and_print_results("Custom Word2Vec Analyzer", df_test['label'], predictions_w2v)
    except Exception as e:
        print(f"\nCustom Word2Vec model failed: {e}")
        print("This can be due to memory issues or a problem with your TensorFlow installation.")
        print("-" * 45)

    # --- 4. Run Experiments (TF-IDF Benchmarks) ---
    print("\n--- Starting Experiments 3 & 4: TF-IDF Benchmarks ---")
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
        X_train_vec = None # Skip next models

    if X_train_vec is not None:
        # Experiment 3: Scikit-learn Multinomial Naive Bayes
        sklearn_nb = MultinomialNB(alpha=0.1)
        sklearn_nb.fit(X_train_vec, df_train['label'])
        predictions_sklearn_nb = sklearn_nb.predict(X_test_vec)
        evaluate_and_print_results("Scikit-learn MultinomialNB (TF-IDF)", df_test['label'], predictions_sklearn_nb)
        
        # Experiment 4: Scikit-learn Logistic Regression (Benchmark)
        log_reg = LogisticRegression(
            random_state=MODEL_PARAMS['random_state_seed'], 
            max_iter=1000
        )
        log_reg.fit(X_train_vec, df_train['label'])
        predictions_log_reg = log_reg.predict(X_test_vec)
        evaluate_and_print_results("Scikit-learn Logistic Regression (TF-IDF)", df_test['label'], predictions_log_reg)

    # --- 5. Run Experiment (Deep Learning) ---
    print("\n--- Starting Experiment 5: Keras Bidirectional LSTM ---")
    try:
        build_and_train_lstm(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"\n--- !!! ERROR !!! ---")
        print(f"LSTM Model Failed: {e}")
        print("This can be due to memory issues or a problem with your TensorFlow installation.")
        print("-" * 45)

if __name__ == '__main__':
    main()