package testdata

import (
	"math/rand"
	"strings"
)

// Text corpus generators for sparse vector benchmarks.
// Each generator produces documents with domain-specific vocabulary and structure.

// CorpusType identifies a text domain.
type CorpusType int

const (
	CorpusAcademic CorpusType = iota
	CorpusCode
	CorpusProduct
	CorpusLegal
	CorpusMultilingual
)

// GenerateCorpus creates count documents from the specified domain.
func GenerateCorpus(corpusType CorpusType, count int, rng *rand.Rand) []string {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	switch corpusType {
	case CorpusAcademic:
		return generateAcademic(count, rng)
	case CorpusCode:
		return generateCode(count, rng)
	case CorpusProduct:
		return generateProduct(count, rng)
	case CorpusLegal:
		return generateLegal(count, rng)
	case CorpusMultilingual:
		return generateMultilingual(count, rng)
	default:
		return generateAcademic(count, rng)
	}
}

var academicNouns = []string{
	"algorithm", "analysis", "architecture", "benchmark", "classification",
	"clustering", "computation", "convergence", "dataset", "dimensionality",
	"distribution", "embedding", "encoder", "evaluation", "extraction",
	"framework", "gradient", "hypothesis", "inference", "kernel",
	"latency", "manifold", "network", "optimization", "parameter",
	"performance", "quantization", "regression", "representation", "scalability",
	"similarity", "sparsity", "tensor", "throughput", "topology",
	"transformer", "validation", "variance", "vector", "workspace",
}

var academicVerbs = []string{
	"achieves", "analyzes", "approximates", "computes", "demonstrates",
	"evaluates", "extends", "generalizes", "implements", "improves",
	"measures", "optimizes", "outperforms", "proposes", "reduces",
	"shows", "surpasses", "trains", "validates", "yields",
}

var academicAdj = []string{
	"approximate", "dense", "distributed", "efficient", "empirical",
	"hierarchical", "hybrid", "iterative", "linear", "multi-scale",
	"non-parametric", "optimal", "parallel", "probabilistic", "robust",
	"scalable", "sparse", "state-of-the-art", "stochastic", "supervised",
}

func generateAcademic(count int, rng *rand.Rand) []string {
	docs := make([]string, count)
	for i := 0; i < count; i++ {
		numSentences := 3 + rng.Intn(8)
		var sb strings.Builder
		for s := 0; s < numSentences; s++ {
			if s > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString("The ")
			sb.WriteString(academicAdj[rng.Intn(len(academicAdj))])
			sb.WriteString(" ")
			sb.WriteString(academicNouns[rng.Intn(len(academicNouns))])
			sb.WriteString(" ")
			sb.WriteString(academicVerbs[rng.Intn(len(academicVerbs))])
			sb.WriteString(" ")
			sb.WriteString(academicAdj[rng.Intn(len(academicAdj))])
			sb.WriteString(" ")
			sb.WriteString(academicNouns[rng.Intn(len(academicNouns))])
			sb.WriteString(".")
		}
		docs[i] = sb.String()
	}
	return docs
}

var codeTokens = []string{
	"func", "return", "error", "nil", "interface", "struct", "map", "slice",
	"goroutine", "channel", "mutex", "context", "handler", "middleware",
	"database", "query", "insert", "update", "delete", "transaction",
	"cache", "index", "search", "filter", "sort", "hash", "tree",
	"config", "logger", "metric", "trace", "span", "endpoint", "route",
	"request", "response", "status", "header", "body", "payload",
	"encode", "decode", "serialize", "marshal", "unmarshal", "parse",
	"validate", "sanitize", "transform", "aggregate", "pipeline",
}

func generateCode(count int, rng *rand.Rand) []string {
	docs := make([]string, count)
	for i := 0; i < count; i++ {
		numTokens := 20 + rng.Intn(80)
		tokens := make([]string, numTokens)
		for t := 0; t < numTokens; t++ {
			tokens[t] = codeTokens[rng.Intn(len(codeTokens))]
		}
		docs[i] = strings.Join(tokens, " ")
	}
	return docs
}

var productNouns = []string{
	"battery", "camera", "charger", "display", "earbuds", "fitness",
	"headphones", "keyboard", "laptop", "microphone", "monitor", "mouse",
	"phone", "printer", "router", "speaker", "storage", "tablet",
	"tracker", "watch",
}

var productAdj = []string{
	"affordable", "compact", "durable", "ergonomic", "fast",
	"lightweight", "portable", "premium", "reliable", "sleek",
	"smart", "versatile", "waterproof", "wireless", "high-quality",
}

func generateProduct(count int, rng *rand.Rand) []string {
	docs := make([]string, count)
	for i := 0; i < count; i++ {
		numSentences := 2 + rng.Intn(4)
		var sb strings.Builder
		for s := 0; s < numSentences; s++ {
			if s > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString("This ")
			sb.WriteString(productAdj[rng.Intn(len(productAdj))])
			sb.WriteString(" ")
			sb.WriteString(productNouns[rng.Intn(len(productNouns))])
			sb.WriteString(" features ")
			sb.WriteString(productAdj[rng.Intn(len(productAdj))])
			sb.WriteString(" design and ")
			sb.WriteString(productAdj[rng.Intn(len(productAdj))])
			sb.WriteString(" performance.")
		}
		docs[i] = sb.String()
	}
	return docs
}

var legalTerms = []string{
	"agreement", "arbitration", "breach", "clause", "compliance",
	"confidentiality", "consent", "damages", "defendant", "disclosure",
	"enforcement", "equity", "fiduciary", "governance", "indemnification",
	"jurisdiction", "liability", "negligence", "obligation", "plaintiff",
	"precedent", "provision", "remedy", "representation", "statute",
	"stipulation", "termination", "tribunal", "warranty", "waiver",
}

func generateLegal(count int, rng *rand.Rand) []string {
	docs := make([]string, count)
	for i := 0; i < count; i++ {
		numClauses := 3 + rng.Intn(6)
		var sb strings.Builder
		for c := 0; c < numClauses; c++ {
			if c > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString("The ")
			sb.WriteString(legalTerms[rng.Intn(len(legalTerms))])
			sb.WriteString(" regarding ")
			sb.WriteString(legalTerms[rng.Intn(len(legalTerms))])
			sb.WriteString(" shall be subject to ")
			sb.WriteString(legalTerms[rng.Intn(len(legalTerms))])
			sb.WriteString(" and ")
			sb.WriteString(legalTerms[rng.Intn(len(legalTerms))])
			sb.WriteString(".")
		}
		docs[i] = sb.String()
	}
	return docs
}

func generateMultilingual(count int, rng *rand.Rand) []string {
	// Mix English + transliterated tokens to simulate multilingual content
	multiTokens := append([]string{}, academicNouns...)
	multiTokens = append(multiTokens,
		"recherche", "algorithme", "analyse", "donnees", "apprentissage",
		"forschung", "algorithmus", "datenbank", "verarbeitung", "ergebnis",
		"investigacion", "algoritmo", "datos", "rendimiento", "resultado",
		"ricerca", "algoritmo", "database", "prestazioni", "risultato",
	)

	docs := make([]string, count)
	for i := 0; i < count; i++ {
		numTokens := 15 + rng.Intn(50)
		tokens := make([]string, numTokens)
		for t := 0; t < numTokens; t++ {
			tokens[t] = multiTokens[rng.Intn(len(multiTokens))]
		}
		docs[i] = strings.Join(tokens, " ")
	}
	return docs
}
