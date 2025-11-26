package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
)

var stopWords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
	"be": true, "by": true, "for": true, "from": true, "has": true, "he": true,
	"in": true, "is": true, "it": true, "its": true, "of": true, "on": true,
	"that": true, "the": true, "to": true, "was": true, "were": true, "will": true,
	"with": true, "i": true, "you": true, "your": true, "me": true, "my": true,
}

var nonAlphaRegex = regexp.MustCompile("[^a-z ]+")

func preprocessText(text string) []string {
	text = strings.ToLower(text)
	text = nonAlphaRegex.ReplaceAllString(text, "")

	tokens := strings.Fields(text)

	filteredTokens := make([]string, 0)
	for _, token := range tokens {
		if !stopWords[token] && len(token) > 2 {
			filteredTokens = append(filteredTokens, token)
		}
	}
	return filteredTokens
}

type NaiveBayesClassifier struct {
	smoothing            float64
	vocab                map[string]bool
	logPrior             map[string]float64
	logLikelihood        map[string]map[string]float64
	logLikelihoodUnknown map[string]float64
	classCounts          map[string]int
	wordCounts           map[string]map[string]int
	mu                   sync.Mutex
}

func NewNaiveBayesClassifier(smoothing float64) *NaiveBayesClassifier {
	return &NaiveBayesClassifier{
		smoothing:            smoothing,
		vocab:                make(map[string]bool),
		logPrior:             make(map[string]float64),
		logLikelihood:        make(map[string]map[string]float64),
		logLikelihoodUnknown: make(map[string]float64),
		classCounts:          make(map[string]int),
		wordCounts: map[string]map[string]int{
			"fake": make(map[string]int),
			"real": make(map[string]int),
		},
	}
}

func (c *NaiveBayesClassifier) Train(documents []string, labels []string) {
	if len(documents) != len(labels) {
		log.Fatal("Documents and labels slices must have the same length")
	}

	totalDocs := len(documents)
	if totalDocs == 0 {
		log.Println("Warning: NaiveBayesClassifier trained on 0 documents.")
		return
	}

	var wg sync.WaitGroup
	wg.Add(totalDocs)

	for i := 0; i < totalDocs; i++ {
		go func(doc, label string) {
			defer wg.Done()
			words := preprocessText(doc)

			c.mu.Lock()
			c.classCounts[label]++
			for _, word := range words {
				c.vocab[word] = true
				c.wordCounts[label][word]++
			}
			c.mu.Unlock()
		}(documents[i], labels[i])
	}

	wg.Wait()

	c.logPrior["fake"] = math.Log(float64(c.classCounts["fake"]) / float64(totalDocs))
	c.logPrior["real"] = math.Log(float64(c.classCounts["real"]) / float64(totalDocs))

	vocabSize := float64(len(c.vocab))
	totalWordsFake := 0
	for _, count := range c.wordCounts["fake"] {
		totalWordsFake += count
	}
	totalWordsReal := 0
	for _, count := range c.wordCounts["real"] {
		totalWordsReal += count
	}

	denomFake := float64(totalWordsFake) + c.smoothing*vocabSize
	denomReal := float64(totalWordsReal) + c.smoothing*vocabSize

	c.logLikelihoodUnknown["fake"] = math.Log(c.smoothing / denomFake)
	c.logLikelihoodUnknown["real"] = math.Log(c.smoothing / denomReal)

	c.logLikelihood["fake"] = make(map[string]float64)
	c.logLikelihood["real"] = make(map[string]float64)

	for word := range c.vocab {
		c.logLikelihood["fake"][word] = math.Log((float64(c.wordCounts["fake"][word]) + c.smoothing) / denomFake)
		c.logLikelihood["real"][word] = math.Log((float64(c.wordCounts["real"][word]) + c.smoothing) / denomReal)
	}
}

func (c *NaiveBayesClassifier) Predict(document string) string {
	words := preprocessText(document)

	logPosteriorFake := c.logPrior["fake"]
	logPosteriorReal := c.logPrior["real"]

	for _, word := range words {
		if ll, ok := c.logLikelihood["fake"][word]; ok {
			logPosteriorFake += ll
		} else {
			logPosteriorFake += c.logLikelihoodUnknown["fake"]
		}

		if ll, ok := c.logLikelihood["real"][word]; ok {
			logPosteriorReal += ll
		} else {
			logPosteriorReal += c.logLikelihoodUnknown["real"]
		}
	}

	if logPosteriorFake > logPosteriorReal {
		return "fake"
	}
	return "real"
}

func loadDataFromCSV(filepath string, label string, textColumnName string) ([]string, []string, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, nil, fmt.Errorf("Error opening file %s: %w", filepath, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var documents []string
	var labels []string

	header, err := reader.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("Error reading header from %s: %w", filepath, err)
	}

	textColumnIndex := -1
	for i, colName := range header {
		if colName == textColumnName {
			textColumnIndex = i
			break
		}
	}

	if textColumnIndex == -1 {
		return nil, nil, fmt.Errorf("Error: text column '%s' not found in %s", textColumnName, filepath)
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("Error reading record from %s: %w", filepath, err)
		}

		if len(record) > textColumnIndex {
			documents = append(documents, record[textColumnIndex])
			labels = append(labels, label)
		}
	}

	return documents, labels, nil
}

// func shuffleData(docs []string, labels []string) {
// 	r := rand.New(rand.NewSource(time.Now().UnixNano()))
// 	r.Shuffle(len(docs), func(i, j int) {
// 		docs[i], docs[j] = docs[j], docs[i]
// 		labels[i], labels[j] = labels[j], labels[i]
// 	})
// }

// func evaluate(predictions, actual []string) {
// 	var truePositive, falsePositive, falseNegative, trueNegative int
// 	posLabel := "fake"
// 	negLabel := "real"

// 	for i := 0; i < len(predictions); i++ {
// 		pred := predictions[i]
// 		act := actual[i]

// 		if pred == posLabel && act == posLabel {
// 			truePositive++
// 		} else if pred == posLabel && act == negLabel {
// 			falsePositive++
// 		} else if pred == negLabel && act == posLabel {
// 			falseNegative++
// 		} else if pred == negLabel && act == negLabel {
// 			trueNegative++
// 		}
// 	}

// 	accuracy := float64(truePositive+trueNegative) / float64(len(predictions))
	
// 	var precision float64
// 	if truePositive+falsePositive > 0 {
// 		precision = float64(truePositive) / float64(truePositive+falsePositive)
// 	}

// 	var recall float64
// 	if truePositive+falseNegative > 0 {
// 		recall = float64(truePositive) / float64(truePositive+falseNegative)
// 	}

// 	var f1 float64
// 	if precision+recall > 0 {
// 		f1 = 2 * (precision * recall) / (precision + recall)
// 	}

// 	fmt.Println("\n--- Evaluation Results ---")
// 	fmt.Printf("  Accuracy:  %.4f\n", accuracy)
// 	fmt.Printf("  Precision: %.4f (for 'fake' class)\n", precision)
// 	fmt.Printf("  Recall:    %.4f (for 'fake' class)\n", recall)
// 	fmt.Printf("  F1-Score:  %.4f (for 'fake' class)\n", f1)
// 	fmt.Println("  Confusion Matrix:")
// 	fmt.Printf("     \t  (Pred Real) (Pred Fake)\n")
// 	fmt.Printf("   (True Real) %5d       %5d\n", trueNegative, falsePositive)
// 	fmt.Printf("   (True Fake) %5d       %5d\n", falseNegative, truePositive)
// 	fmt.Println("--------------------------")
// }

var classifier *NaiveBayesClassifier

type PredictionRequest struct {
	Text string `json:"text"`
}

type PredictionResponse struct {
	IsFake int `json:"is_fake"`
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var request PredictionRequest
	err := json.NewDecoder(r.Body).Decode(&request)
	if err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if request.Text == "" {
		http.Error(w, "'text' field is required", http.StatusBadRequest)
		return
	}

	prediction := classifier.Predict(request.Text)

	response := PredictionResponse{}
	if prediction == "fake" {
		response.IsFake = 1
	} else {
		response.IsFake = 0
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	dataFileReal := "True.csv"
	dataFileFake := "Fake.csv"
	textColumn := "text"

	fmt.Println("Loading data from CSV files...")
	realDocs, realLabels, err := loadDataFromCSV(dataFileReal, "real", textColumn)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d real documents.\n", len(realDocs))

	fakeDocs, fakeLabels, err := loadDataFromCSV(dataFileFake, "fake", textColumn)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d fake documents.\n", len(fakeDocs))

	docs := append(realDocs, fakeDocs...)
	labels := append(realLabels, fakeLabels...)

	if len(docs) == 0 {
		log.Fatal("No documents were loaded. Please check CSV paths and column name.")
	}

	fmt.Printf("Total documents loaded: %d\n", len(docs))

	classifier = NewNaiveBayesClassifier(1.0)

	fmt.Println("\nTraining classifier on the full dataset (this may take a moment)...")
	classifier.Train(docs, labels)
	fmt.Println("Training complete.")

	http.HandleFunc("/predict", predictHandler)

	port := ":8080"
	fmt.Printf("\nServer starting on port %s...\n", port)
	fmt.Println("Endpoint available at POST http://localhost:8080/predict")
	log.Fatal(http.ListenAndServe(port, nil))
}

