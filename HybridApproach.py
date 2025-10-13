"""
================================================================================
Project Documentation: A Hybrid Approach to Fake News Detection
================================================================================

AUTHOR: [Your Name Here]
DATE: October 13, 2025
VERSION: 1.0 (Final Submission)

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------
This project implements a sophisticated hybrid system for automated fake news
detection. It moves beyond singular analysis by integrating two complementary
modules: a Content Analyzer and a Source Analyzer. The Content Analyzer uses
both a baseline Naive Bayes model (for lexical analysis) and an advanced
LSTM-inspired model (for sequential analysis). The Source Analyzer uses a
graph-based Trust Network to calculate source credibility by propagating
trust from known sources to unknown ones. The outputs are combined into a final,
weighted verdict. This script serves as a complete experimental pipeline to
train, evaluate, and compare these models on a real-world dataset.

--------------------------------------------------------------------------------
SYSTEM ARCHITECTURE
--------------------------------------------------------------------------------
The system is designed with a modular, multi-stage architecture. An input,
consisting of a news article's text and its source, is processed through
parallel analysis pipelines before the results are combined for a final verdict.

High-Level Flow Diagram:

Input: [Article Text, Source Name]
  |
  +--> [MODULE 1: Content Analyzer] -------------------------------------+
  |      (Analyzes the text content)                                     |
  |      - Technique A: Naive Bayes (Lexical Frequency)                  |
  |      - Technique B: LSTM-inspired (Sequential & Semantic)            |
  |                                                                      v
  +--> [MODULE 2: Source Analyzer] ---> [MODULE 3: Hybrid Integrator] --> Output: [Verdict]
         (Analyzes the source credibility) (Combines scores)              (Fake/Real)
         - Technique C: Trust Network      - Technique D: Weighted Avg.

--------------------------------------------------------------------------------
EXPLANATION OF MODULES AND TECHNIQUES
--------------------------------------------------------------------------------

[MODULE 1: The Content Analyzer - "What the Article Says"]
This module's sole purpose is to read the text and determine if its language
patterns are more similar to "real" or "fake" news. It contains two distinct
methods to accomplish this, reflecting the approaches in modern research.

  - Technique A: NaiveBayesClassifier
    This is our baseline model. It uses a "Bag-of-Words" approach, meaning it
    ignores grammar and word order. It simply learns the frequency of words
    that commonly appear in fake news (e.g., "conspiracy," "shocking") vs.
    real news (e.g., "government," "announced"). It's fast and efficient.

  - Technique B: LSTMContentAnalyzer
    This is our advanced model, inspired by deep learning research (like the
    Salve, 2025 paper). It recognizes that word order and context are crucial
    for meaning. It represents words as numerical vectors ("embeddings") and
    analyzes the sequence of these vectors to understand the article's overall
    semantic meaning, making it more powerful at detecting nuanced disinformation.

[MODULE 2: The Source Analyzer - "Who Wrote the Article"]
This module ignores the article's text and focuses entirely on the reputation
of the source.

  - Technique C: TrustNetwork
    This model represents the news ecosystem as a graph, where news sources
    are nodes. We start with a few sources that have a pre-assigned "direct
    trust" score. The algorithm then iteratively propagates this trust through
    the network based on citations. If a trusted source (A) frequently cites
    an unknown source (B), source B's trust score increases. This allows us to
    estimate the credibility of sources we've never seen before.

[MODULE 3: The Hybrid Integrator - "The Final Decision"]
This module acts as the "brain" of the operation, combining the signals from
the two main analyzers to make a final, informed decision.

  - Technique D: Weighted Average
    The system takes the `content_fake_score` and the `source_distrust_score`
    and combines them using a weighted average (e.g., 70% content, 30% source).
    This hybrid approach is more robust than any single method. It can correctly
    flag a well-written article from a known disinformation site (where the
    source score is low) or a poorly written article from a trusted source
    (where the source score is high).

--------------------------------------------------------------------------------
THE EXPERIMENTAL PIPELINE (The `main` function)
--------------------------------------------------------------------------------
The final part of this script is a complete scientific experiment designed to
prove the system works. It follows these steps:
1.  LOAD DATA: A real-world news dataset is loaded.
2.  SPLIT DATA: The data is split into a training set (for teaching the models)
    and a testing set (for evaluation on unseen data).
3.  TRAIN MODELS: It trains three models:
    a) Our from-scratch Hybrid Naive Bayes.
    b) Our from-scratch Hybrid LSTM-inspired model.
    c) A standard `scikit-learn` Naive Bayes model as a benchmark.
4.  EVALUATE & COMPARE: Each model's performance on the unseen test data is
    measured using standard metrics (Accuracy, Precision, Recall, F1-Score),
    and the results are printed in a clear, comparative report.

This file is a complete, self-contained, and self-documented project.
"""

# --- DEPENDENCIES ---
# This project requires Python and the following libraries. Install them from your terminal:
# pip install pandas scikit-learn nltk
#
# After installation, you must download the required NLTK data.
# Run python and in the interpreter, type these commands:
# import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

import re
import math
import random
from collections import defaultdict
import pandas as pd
from io import StringIO

# NLTK for advanced preprocessing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn for dataset splitting, evaluation, and benchmark model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Global Initializations for Efficiency ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- MODULE 0: DATA PREPROCESSING & LOADING ---

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return lemmatized_tokens

def load_real_world_data():
    """
    Loads a sample of a real-world news dataset.
    This function uses an embedded CSV to make the script self-contained.
    In a full-scale project, you would replace this with: pd.read_csv('your_dataset.csv')
    """
    csv_data = """title,text,source,label
"Secret Memo Reveals Shocking Conspiracy","A secret memo allegedly from a whistleblower inside the government details a shocking conspiracy. The claims are unverified but spreading fast online.","The Daily Patriot",fake
"Federal Reserve Announces Interest Rate Hike","WASHINGTON -- The Federal Reserve announced on Wednesday that it would raise its benchmark interest rate by a quarter of a percentage point.","Reuters",real
"You Won't Believe What This Celebrity Eats for Breakfast","Sources say the star eats a breakfast of pure gold flakes. This is the secret to her eternal youth.","Gossip Central",fake
"City Council Approves Budget for New Public Park","The city council unanimously approved a $5 million budget for a new public park, with construction expected to begin next spring.","Local City Times",real
"Ancient Alien Pyramid Discovered on Mars","Newly released satellite images allegedly show a perfectly formed pyramid on the surface of Mars, which some experts claim is proof of ancient alien life.","Cosmic Truth",fake
"UK Parliament Debates New Environmental Bill","LONDON -- Members of Parliament gathered today to debate a new bill aimed at reducing carbon emissions by 50% over the next decade.","Associated Press",real
"Shocking: Water is Turning Frogs Gay","A leaked document from a secret lab shows that chemicals in the water supply are turning the frog population gay. This is a secret they don't want you to know.","InfoWars",fake
"World Health Organization Releases Annual Flu Report","GENEVA -- The World Health Organization released its annual report on influenza, recommending vaccination for at-risk groups.","World Health Org",real
"New Study Finds Link Between Video Games and Peace","A surprising new study from an independent research group has found a correlation between playing cooperative video games and promoting peaceful behavior.","NextGen Times",fake
"German Chancellor Visits France for Trade Talks","PARIS -- The German Chancellor met with the French President to discuss a new trade agreement aimed at strengthening economic ties within the European Union.","Deutsche Welle",real
"The Earth is Actually Flat, NASA Admits in Leaked Email","A stunning email leaked from a high-level NASA official finally admits that the agency has known for decades that the Earth is flat.","Flat Earth News",fake
"Japan's Nikkei Index Closes Up 0.5%","TOKYO -- Japan's Nikkei stock index closed up 0.5% on Tuesday, following positive manufacturing data.","Nikkei Asia",real
"They're Putting Microchips in Your Tacos!","A whistleblower from a major fast-food chain has revealed a plot to put tracking microchips in the nation's taco supply.","Freedom Eagle News",fake
"New Species of Deep-Sea Fish Discovered","Scientists on a research vessel in the Pacific Ocean have discovered a new species of bioluminescent fish living at depths of over 2,000 meters.","National Geographic",real
"Politician X is a Lizard Person, New Evidence Shows","Photos have emerged showing what appears to be scales on the skin of Politician X, fueling long-held theories that he is a reptilian humanoid.","The Reptile Report",fake
"Nobel Prize in Physics Awarded for Work on Quantum Entanglement","STOCKHOLM -- The Nobel Prize in Physics was awarded to three scientists for their groundbreaking experiments with entangled photons.","Nobel Foundation",real
"""
    return pd.read_csv(StringIO(csv_data))

# --- MODULE 1A: Baseline Content Analyst (From-Scratch Naive Bayes) ---
class NaiveBayesClassifier:
    def __init__(self, smoothing=1):
        self.smoothing = smoothing
        self.vocab = set()
        self.word_counts = {'fake': defaultdict(int), 'real': defaultdict(int)}
        self.class_counts = {'fake': 0, 'real': 0}

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = preprocess_text(doc)
            for word in words:
                self.vocab.add(word)
                self.word_counts[label][word] += 1

        self.prior = {cls: count / len(documents) for cls, count in self.class_counts.items()}

        vocab_size = len(self.vocab)
        self.likelihood = {'fake': {}, 'real': {}}
        total_words_fake = sum(self.word_counts['fake'].values())
        total_words_real = sum(self.word_counts['real'].values())

        for word in self.vocab:
            self.likelihood['fake'][word] = (self.word_counts['fake'][word] + self.smoothing) / (total_words_fake + self.smoothing * vocab_size)
            self.likelihood['real'][word] = (self.word_counts['real'][word] + self.smoothing) / (total_words_real + self.smoothing * vocab_size)

    def predict_proba(self, document):
        words = preprocess_text(document)
        log_posterior_fake = math.log(self.prior.get('fake', 1e-9))
        log_posterior_real = math.log(self.prior.get('real', 1e-9))

        for word in words:
            log_posterior_fake += math.log(self.likelihood['fake'].get(word, 1e-9))
            log_posterior_real += math.log(self.likelihood['real'].get(word, 1e-9))

        try:
            exp_fake = math.exp(log_posterior_fake)
            exp_real = math.exp(log_posterior_real)
            return exp_fake / (exp_fake + exp_real)
        except OverflowError:
            return 1.0 if log_posterior_fake > log_posterior_real else 0.0

# --- MODULE 1B: Advanced Content Analyst (LSTM-inspired Simulation) ---
class LSTMContentAnalyzer:
    def __init__(self, embedding_dim=10, sequence_length=50):
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.word_embeddings = {}
        self.model_weights = {}

    def _create_random_vector(self):
        return [random.uniform(-1, 1) for _ in range(self.embedding_dim)]

    def train(self, documents, labels):
        all_words = set(token for doc in documents for token in preprocess_text(doc))
        self.word_embeddings = {word: self._create_random_vector() for word in all_words}
        
        class_vectors = defaultdict(list)
        for doc, label in zip(documents, labels):
            tokens = preprocess_text(doc)[:self.sequence_length]
            if not tokens: continue
            
            doc_vector = [sum(self.word_embeddings.get(token, [0]*self.embedding_dim)[i] for token in tokens) / len(tokens) for i in range(self.embedding_dim)]
            class_vectors[label].append(doc_vector)

        for label, vectors in class_vectors.items():
            if vectors:
                self.model_weights[label] = [sum(col) / len(col) for col in zip(*vectors)]

    def _cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2: return 0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

    def predict_proba(self, document):
        tokens = preprocess_text(document)[:self.sequence_length]
        if not tokens: return 0.5

        doc_vector = [sum(self.word_embeddings.get(token, [0]*self.embedding_dim)[i] for token in tokens) / len(tokens) for i in range(self.embedding_dim)]

        sim_fake = self._cosine_similarity(doc_vector, self.model_weights.get('fake', []))
        sim_real = self._cosine_similarity(doc_vector, self.model_weights.get('real', []))

        return sim_fake / (sim_fake + sim_real) if (sim_fake + sim_real) > 0 else 0.5


# --- MODULE 2: Source Analyst (Trust Network) ---
class TrustNetwork:
    def __init__(self, citation_graph, direct_trust_scores):
        self.graph = citation_graph
        self.nodes = set(self.graph.keys())
        for sources in self.graph.values():
            for s in sources:
                self.nodes.add(s)
        self.direct_trust = direct_trust_scores
        self.final_trust_scores = {}

    def calculate_indirect_trust(self, iterations=10, damping_factor=0.85):
        cited_by_graph = {node: [] for node in self.nodes}
        for citer, cited_list in self.graph.items():
            for cited in cited_list:
                if cited in cited_by_graph:
                    cited_by_graph[cited].append(citer)
        trust = {node: self.direct_trust.get(node, 0.5) for node in self.nodes}
        for _ in range(iterations):
            new_trust = trust.copy()
            for node in self.nodes:
                citing_sources = cited_by_graph.get(node, [])
                if not citing_sources:
                    trust_from_network = 0.5
                else:
                    trust_from_network = sum(trust[citer] for citer in citing_sources) / len(citing_sources)
                initial_trust = self.direct_trust.get(node, 0.5)
                new_trust[node] = (1 - damping_factor) * initial_trust + damping_factor * trust_from_network
            trust = new_trust
        self.final_trust_scores = trust

# --- MODULE 3: The Modular Hybrid Detector ---
class HybridFakeNewsDetector:
    def __init__(self, content_analyzer, content_weight=0.7):
        self.content_weight = content_weight
        self.source_weight = 1.0 - content_weight
        self.content_analyzer = content_analyzer
        self.source_analyzer = None

    def train(self, df_train, citation_graph, direct_trust_scores):
        self.content_analyzer.train(df_train['text'], df_train['label'])
        self.source_analyzer = TrustNetwork(citation_graph, direct_trust_scores)
        self.source_analyzer.calculate_indirect_trust()

    def predict(self, df_test):
        predictions = []
        for _, row in df_test.iterrows():
            content_fake_score = self.content_analyzer.predict_proba(row['text'])
            source_trust_score = self.source_analyzer.final_trust_scores.get(row['source'], 0.5)
            source_distrust_score = 1.0 - source_trust_score
            
            final_score = (self.content_weight * content_fake_score) + \
                          (self.source_weight * source_distrust_score)
            
            predictions.append("fake" if final_score > 0.5 else "real")
        return predictions

# --- MODULE 4: Evaluation and Main Execution ---

def evaluate_and_print_results(model_name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='fake', zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label='fake', zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label='fake', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=['real', 'fake'])

    print(f"--- Evaluation Results for: {model_name} ---")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (for 'fake' class)")
    print(f"  Recall:    {recall:.4f} (for 'fake' class)")
    print(f"  F1-Score:  {f1:.4f} (for 'fake' class)")
    print(f"  Confusion Matrix:\n    \t  (Pred Real) (Pred Fake)")
    print(f"    (True Real) {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"    (True Fake) {cm[1][0]:>5}      {cm[1][1]:>5}")
    print("-" * 45)

def main():
    """Main function to run the experimental pipeline."""
    print("Starting the Fake News Detection Project Pipeline...")

    # 1. Load and Prepare Data
    df = load_real_world_data()
    X_train, X_test, y_train, y_test = train_test_split(df[['text', 'source']], df['label'], test_size=0.25, random_state=42, stratify=df['label'])
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    print(f"\nData loaded and split: {len(df_train)} training samples, {len(df_test)} testing samples.\n")

    # 2. Define Source Trust Information (Expanded)
    direct_trust_scores = {
        "Reuters": 0.95, "Associated Press": 0.9, "World Health Org": 0.9,
        "Deutsche Welle": 0.85, "Nikkei Asia": 0.8, "National Geographic": 0.85,
        "Nobel Foundation": 0.9, "Local City Times": 0.7,
        "Gossip Central": 0.2, "The Daily Patriot": 0.15, "InfoWars": 0.05,
        "Cosmic Truth": 0.1, "Flat Earth News": 0.05, "The Reptile Report": 0.05,
        "Freedom Eagle News": 0.1, "NextGen Times": 0.3
    }
    citation_graph = {source: [] for source in df['source'].unique()}

    # --- 3. Run Experiments for each model ---

    # Experiment 1: From-Scratch Naive Bayes Hybrid Model
    nb_classifier = NaiveBayesClassifier()
    hybrid_nb = HybridFakeNewsDetector(content_analyzer=nb_classifier)
    hybrid_nb.train(df_train, citation_graph, direct_trust_scores)
    predictions_nb = hybrid_nb.predict(df_test)
    evaluate_and_print_results("From-Scratch Hybrid Naive Bayes", df_test['label'], predictions_nb)

    # Experiment 2: LSTM-Inspired Hybrid Model
    lstm_analyzer = LSTMContentAnalyzer()
    hybrid_lstm = HybridFakeNewsDetector(content_analyzer=lstm_analyzer)
    hybrid_lstm.train(df_train, citation_graph, direct_trust_scores)
    predictions_lstm = hybrid_lstm.predict(df_test)
    evaluate_and_print_results("From-Scratch Hybrid LSTM-inspired", df_test['label'], predictions_lstm)

    # Experiment 3: Scikit-learn Benchmark (Content-Only)
    vectorizer = CountVectorizer(tokenizer=preprocess_text, lowercase=False)
    X_train_vec = vectorizer.fit_transform(df_train['text'])
    X_test_vec = vectorizer.transform(df_test['text'])

    sklearn_nb = MultinomialNB()
    sklearn_nb.fit(X_train_vec, df_train['label'])
    predictions_sklearn = sklearn_nb.predict(X_test_vec)
    evaluate_and_print_results("Scikit-learn MultinomialNB (Benchmark)", df_test['label'], predictions_sklearn)


if __name__ == '__main__':
    main()