import re
from collections import defaultdict
import math

# --- MODULE 1: The Content Analyst ---
class NaiveBayesClassifier:
    """
    A Naive Bayes Classifier for text classification.
    This component analyzes the content of the news article.
    """
    def __init__(self, smoothing=1):
        self.smoothing = smoothing
        self.vocab = set()
        self.word_counts = {'fake': defaultdict(int), 'real': defaultdict(int)}
        self.class_counts = {'fake': 0, 'real': 0}
        self.total_docs = 0
        self.prior = {'fake': 0.0, 'real': 0.0}
        self.likelihood = {'fake': defaultdict(float), 'real': defaultdict(float)}

    def _preprocess(self, text):
        stopwords = {'a', 'about', 'above', 'after', 'again', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'had', 'has', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'if', 'in', 'is', 'it', 'its', 'just', 'me', 'more', 'most', 'my', 'no', 'nor', 'not', 'of', 'off', 'on', 'or', 'our', 'out', 'over', 's', 'same', 'she', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'why', 'with', 'you', 'your'}
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        return [word for word in tokens if word not in stopwords and len(word) > 2]

    def train(self, documents, labels):
        self.total_docs = len(documents)
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = self._preprocess(doc)
            for word in words:
                self.vocab.add(word)
                self.word_counts[label][word] += 1

        self.prior['fake'] = self.class_counts['fake'] / self.total_docs
        self.prior['real'] = self.class_counts['real'] / self.total_docs

        vocab_size = len(self.vocab)
        total_words_fake = sum(self.word_counts['fake'].values())
        total_words_real = sum(self.word_counts['real'].values())

        for word in self.vocab:
            count_fake = self.word_counts['fake'][word]
            self.likelihood['fake'][word] = (count_fake + self.smoothing) / (total_words_fake + self.smoothing * vocab_size)
            count_real = self.word_counts['real'][word]
            self.likelihood['real'][word] = (count_real + self.smoothing) / (total_words_real + self.smoothing * vocab_size)

    def predict(self, document):
        words = self._preprocess(document)
        log_posterior_fake = math.log(self.prior.get('fake', 1e-9))
        log_posterior_real = math.log(self.prior.get('real', 1e-9))

        for word in words:
            if word in self.vocab:
                log_posterior_fake += math.log(self.likelihood['fake'].get(word, 1e-9))
                log_posterior_real += math.log(self.likelihood['real'].get(word, 1e-9))
        
        # Normalize to get a probability-like score using softmax
        exp_fake = math.exp(log_posterior_fake)
        exp_real = math.exp(log_posterior_real)
        prob_fake = exp_fake / (exp_fake + exp_real)
        
        return prob_fake

# --- MODULE 2: The Source Analyst ---
class TrustNetwork:
    """
    Manages the trust scores of news sources based on their network connections.
    """
    def __init__(self, citation_graph, direct_trust_scores):
        self.graph = citation_graph
        self.nodes = set(self.graph.keys())
        for sources in self.graph.values():
            for s in sources:
                self.nodes.add(s)

        self.direct_trust = direct_trust_scores
        self.final_trust_scores = {}

    def calculate_indirect_trust(self, iterations=10, damping_factor=0.85):
        """
        Calculates trust scores iteratively, propagating them through the network.
        """
        cited_by_graph = {node: [] for node in self.nodes}
        for citer, cited_list in self.graph.items():
            for cited in cited_list:
                if cited in cited_by_graph:
                    cited_by_graph[cited].append(citer)

        trust = {node: self.direct_trust.get(node, 0.5) for node in self.nodes}

        for _ in range(iterations):
            new_trust = trust.copy()
            for node in self.nodes:
                citing_sources = cited_by_graph[node]
                if not citing_sources:
                    trust_from_network = 0.5
                else:
                    trust_from_network = sum(trust[citer] for citer in citing_sources) / len(citing_sources)

                initial_trust = self.direct_trust.get(node, 0.5)
                new_trust[node] = (1 - damping_factor) * initial_trust + damping_factor * trust_from_network
            trust = new_trust
        
        self.final_trust_scores = trust


# --- MODULE 3: The Hybrid Detector ---
class HybridFakeNewsDetector:
    """
    Combines content analysis and source trust analysis for a final verdict.
    """
    def __init__(self, content_weight=0.7, source_weight=0.3):
        self.content_weight = content_weight
        self.source_weight = source_weight
        self.content_analyzer = NaiveBayesClassifier()
        self.source_analyzer = None

    def train(self, documents, labels, citation_graph, direct_trust_scores):
        print("--- Starting Hybrid Model Training ---")
        self.content_analyzer.train(documents, labels)
        print("Content analyzer trained.")
        self.source_analyzer = TrustNetwork(citation_graph, direct_trust_scores)
        self.source_analyzer.calculate_indirect_trust()
        print("Source analyzer trained.")
        print("--- Training Complete ---\n")

    def predict(self, document_text, source):
        content_fake_score = self.content_analyzer.predict(document_text)
        source_trust_score = self.source_analyzer.final_trust_scores.get(source, 0.5)
        source_distrust_score = 1.0 - source_trust_score
        
        final_score = (self.content_weight * content_fake_score) + \
                      (self.source_weight * source_distrust_score)
        
        verdict = "fake" if final_score > 0.5 else "real"
        return {
            "verdict": verdict,
            "final_fake_score": final_score,
            "content_fake_score": content_fake_score,
            "source_trust_score": source_trust_score,
            "source": source
        }

# --- MODULE 4: Test Suite and Main Execution ---
def run_tests():
    """
    Sets up a controlled environment and runs a series of test cases
    to validate the HybridFakeNewsDetector.
    """
    print("--- INITIALIZING TEST ENVIRONMENT ---")
    
    training_data = [
        {"text": "Millionaire celebrity reveals secret to unlimited wealth", "label": "fake", "source": "NewsForYou"},
        {"text": "Scientific study confirms aliens are living among us", "label": "fake", "source": "GlobalTruth"},
        {"text": "New cure for cancer discovered by lone researcher, big pharma is hiding it", "label": "fake", "source": "HealthWatch"},
        {"text": "Local council approves new budget for city parks", "label": "real", "source": "CityTimes"},
        {"text": "Stock market sees slight increase after federal reserve announcement", "label": "real", "source": "Reuters"},
        {"text": "International trade agreement reached between two nations", "label": "real", "source": "AssociatedPress"},
    ]
    
    docs = [d['text'] for d in training_data]
    labels = [d['label'] for d in training_data]

    citation_graph = {
        "NewsForYou": ["GlobalTruth"],
        "GlobalTruth": ["HealthWatch", "ShadowDispatch"],
        "CityTimes": ["Reuters", "AssociatedPress"],
        "Reuters": ["AssociatedPress", "CredibleChronicle"],
        "AssociatedPress": ["Reuters"],
        "HealthWatch": [], "CredibleChronicle": [], "ShadowDispatch": []
    }

    direct_trust_scores = {
        "Reuters": 0.95, "AssociatedPress": 0.9, "CityTimes": 0.75,
        "GlobalTruth": 0.1, "NewsForYou": 0.2
    }
    
    detector = HybridFakeNewsDetector(content_weight=0.7, source_weight=0.3)
    detector.train(docs, labels, citation_graph, direct_trust_scores)

    print("\n--- STARTING TEST CASES ---\n")

    # Test Case 1: Congruent Signals
    print("--- Test Case 1: Congruent Signals ---")
    article1 = {"text": "President gives speech on economic policy", "source": "Reuters"}
    result1 = detector.predict(article1["text"], article1["source"])
    print(f"Test 1A: Real-sounding content from a HIGH-trust source ('{article1['source']}')")
    print(f"  -> EXPECTED: REAL. ACTUAL: {result1['verdict'].upper()}. Score: {result1['final_fake_score']:.2f}\n")
    
    article2 = {"text": "You won't believe what this politician said, shocking secrets revealed", "source": "NewsForYou"}
    result2 = detector.predict(article2["text"], article2["source"])
    print(f"Test 1B: Fake-sounding content from a LOW-trust source ('{article2['source']}')")
    print(f"  -> EXPECTED: FAKE. ACTUAL: {result2['verdict'].upper()}. Score: {result2['final_fake_score']:.2f}\n")

    # Test Case 2: Conflicting Signals
    print("--- Test Case 2: Conflicting Signals ---")
    article3 = {"text": "Shocking celebrity secret revealed by an unknown insider", "source": "Reuters"}
    result3 = detector.predict(article3["text"], article3["source"])
    print(f"Test 2A: Fake-sounding content from a HIGH-trust source ('{article3['source']}')")
    print(f"  -> EXPECTED: FAKE (due to high content weight). ACTUAL: {result3['verdict'].upper()}. Score: {result3['final_fake_score']:.2f}\n")

    article4 = {"text": "City council meeting agenda for Tuesday was released today", "source": "GlobalTruth"}
    result4 = detector.predict(article4["text"], article4["source"])
    print(f"Test 2B: Real-sounding content from a LOW-trust source ('{article4['source']}')")
    print(f"  -> EXPECTED: REAL (due to innocent content). ACTUAL: {result4['verdict'].upper()}. Score: {result4['final_fake_score']:.2f}\n")

    # Test Case 3: Indirect Trust
    print("--- Test Case 3: Indirect Trust ---")
    article5 = {"text": "Official government report on infrastructure published this morning", "source": "CredibleChronicle"}
    result5 = detector.predict(article5["text"], article5["source"])
    print(f"Test 3A: Real content from a source with high INDIRECT trust ('{article5['source']}')")
    print(f"  -> Source Trust: {result5['source_trust_score']:.2f} (should be high)")
    print(f"  -> EXPECTED: REAL. ACTUAL: {result5['verdict'].upper()}. Score: {result5['final_fake_score']:.2f}\n")

    article6 = {"text": "The secret they don't want you to know about the government", "source": "ShadowDispatch"}
    result6 = detector.predict(article6["text"], article6["source"])
    print(f"Test 3B: Fake content from a source with low INDIRECT trust ('{article6['source']}')")
    print(f"  -> Source Trust: {result6['source_trust_score']:.2f} (should be low)")
    print(f"  -> EXPECTED: FAKE. ACTUAL: {result6['verdict'].upper()}. Score: {result6['final_fake_score']:.2f}\n")

    # Test Case 4: Unknown Source
    print("--- Test Case 4: Unknown Source ---")
    article7 = {"text": "Local weather forecast predicts sun for the weekend", "source": "MyLocalBlog"}
    result7 = detector.predict(article7["text"], article7["source"])
    print(f"Test 4: Mundane content from a completely UNKNOWN source ('{article7['source']}')")
    print(f"  -> Source Trust: {result7['source_trust_score']:.2f} (should be default 0.5)")
    print(f"  -> EXPECTED: REAL (decision based on content). ACTUAL: {result7['verdict'].upper()}. Score: {result7['final_fake_score']:.2f}\n")

    print("--- ALL TEST CASES COMPLETE ---")

if __name__ == '__main__':
    run_tests()
