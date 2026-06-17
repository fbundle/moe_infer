// Three-axis reasoning bench, Go port of moe_infer/helpers/bench_axes.py.
//
// Hooks frozen subsets + OpenAI-compatible API into the same verify-and-retry
// harness as the Python version. Streams full per-attempt records to JSONL.
//
//   go run bench.go --model vibethinker-3b-q4 --concurrency 2 --n 10 \
//                    --budget-hint --max-retries 5 --top-p 0.95 \
//                    --max-tokens 40960 --dump-dir ../data/bench_runs/test
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

// ── Shared types ──────────────────────────────────────────────────────────

// Example: one bench question loaded from the frozen subset.
type Example struct {
	ID     string          `json:"id"`
	Prompt string          `json:"prompt"`
	Gold   json.RawMessage `json:"gold"`
	Meta   map[string]any  `json:"meta"`
}

// Completion: model response from one API call.
type Completion struct {
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content"`
	FinishReason     string `json:"finish_reason"`
	TokensUsed       int    `json:"tokens_used"`
}

// VerifyLoopOutcome: result of verify_loop's sequential retry.
type VerifyLoopOutcome struct {
	Comp       *Completion                       // last completion (nil if every call raised)
	Attempts   int                               // 1 = first try worked
	Parsed     any                               // validated value, or nil
	FinalError string                            // last error if failed
	Truncated  bool                              // finish_reason == "length"
	Messages   []openai.ChatCompletionMessage    // full final conversation
	History    []*Completion                     // every call's completion — record-everything
}

// Result: scored outcome for one example, written to JSONL.
type Result struct {
	ID        string          `json:"id"`
	Prompt    string          `json:"prompt"`
	Gold      json.RawMessage `json:"gold"`
	Response  string          `json:"response"`
	Reasoning string          `json:"reasoning"`
	Parsed    any             `json:"parsed"`
	Correct   bool            `json:"correct"`
	Elapsed   float64         `json:"elapsed"`
	Error     string          `json:"error"`
	Meta      map[string]any  `json:"meta"`
	Extras    map[string]any  `json:"extras"`
}

// ── Bench interface + three implementations ───────────────────────────────

// Bench: one axis (ZebraLogic / CLadder / GPQA-Diamond).
type Bench interface {
	Name() string
	SystemPrompt() string
	Schema() *jsonschema.Definition // strict JSON schema for response_format
	Validate(content string) (parsed any, err error)
	Score(parsed any, gold json.RawMessage) (correct bool, extras map[string]any)
}

// ZebraLogic: grid-mode logic puzzles.
type ZebraLogic struct{}
type ZebraOutput struct {
	Header []string   `json:"header"`
	Rows   [][]string `json:"rows"`
}

func (ZebraLogic) Name() string { return "zebralogic" }
func (ZebraLogic) SystemPrompt() string {
	return ("You solve logic-grid puzzles. Respond with a single JSON object " +
		"matching this schema (no markdown, no commentary, just JSON):\n" +
		`  {"header": ["House", "<attr1>", "<attr2>", ...],` + "\n" +
		`   "rows": [["1", "...", ...], ["2", "...", ...], ...]}` + "\n" +
		"Use the exact attribute values from the puzzle.")
}
func (ZebraLogic) Schema() *jsonschema.Definition {
	return &jsonschema.Definition{
		Type:     jsonschema.Object,
		Required: []string{"header", "rows"},
		Properties: map[string]jsonschema.Definition{
			"header": {Type: jsonschema.Array, Items: &jsonschema.Definition{Type: jsonschema.String}},
			"rows":   {Type: jsonschema.Array, Items: &jsonschema.Definition{Type: jsonschema.Array, Items: &jsonschema.Definition{Type: jsonschema.String}}},
		},
	}
}
func (ZebraLogic) Validate(content string) (any, error) {
	var out ZebraOutput
	if err := json.Unmarshal([]byte(content), &out); err != nil {
		return nil, err
	}
	return out, nil
}
func (ZebraLogic) Score(parsed any, gold json.RawMessage) (bool, map[string]any) {
	var goldZ ZebraOutput
	_ = json.Unmarshal(gold, &goldZ)
	p, ok := parsed.(ZebraOutput)
	totalCells := 0
	for _, r := range goldZ.Rows {
		totalCells += len(r)
	}
	if !ok {
		return false, map[string]any{"cell_correct": 0, "cell_total": totalCells}
	}
	correctCells := 0
	for i, gr := range goldZ.Rows {
		if i >= len(p.Rows) {
			break
		}
		for j, gc := range gr {
			if j >= len(p.Rows[i]) {
				break
			}
			if strings.EqualFold(strings.TrimSpace(gc), strings.TrimSpace(p.Rows[i][j])) {
				correctCells++
			}
		}
	}
	all := correctCells == totalCells && len(goldZ.Rows) == len(p.Rows)
	return all, map[string]any{"cell_correct": correctCells, "cell_total": totalCells}
}

// CLadder: causal yes/no.
type CLadder struct{}
type CladderOutput struct {
	Answer string `json:"answer"`
}

func (CLadder) Name() string { return "cladder" }
func (CLadder) SystemPrompt() string {
	return ("Answer the causal-reasoning question. Respond with a single JSON " +
		`object: {"answer": "yes"} or {"answer": "no"}.`)
}
func (CLadder) Schema() *jsonschema.Definition {
	return &jsonschema.Definition{
		Type:     jsonschema.Object,
		Required: []string{"answer"},
		Properties: map[string]jsonschema.Definition{
			"answer": {Type: jsonschema.String, Enum: []string{"yes", "no"}},
		},
	}
}
func (CLadder) Validate(content string) (any, error) {
	var out CladderOutput
	if err := json.Unmarshal([]byte(content), &out); err != nil {
		return nil, err
	}
	if out.Answer != "yes" && out.Answer != "no" {
		return nil, fmt.Errorf("answer must be yes or no, got %q", out.Answer)
	}
	return out, nil
}
func (CLadder) Score(parsed any, gold json.RawMessage) (bool, map[string]any) {
	p, ok := parsed.(CladderOutput)
	if !ok {
		return false, nil
	}
	var g string
	_ = json.Unmarshal(gold, &g)
	return strings.EqualFold(strings.TrimSpace(p.Answer), strings.TrimSpace(g)), nil
}

// GPQADiamond: science MC (A/B/C/D).
type GPQADiamond struct{}
type GpqaOutput struct {
	Answer string `json:"answer"`
}

func (GPQADiamond) Name() string { return "gpqa_diamond" }
func (GPQADiamond) SystemPrompt() string {
	return ("Answer the multiple-choice question. Respond with a single JSON " +
		`object: {"answer": "A"} or "B", "C", "D".`)
}
func (GPQADiamond) Schema() *jsonschema.Definition {
	return &jsonschema.Definition{
		Type:     jsonschema.Object,
		Required: []string{"answer"},
		Properties: map[string]jsonschema.Definition{
			"answer": {Type: jsonschema.String, Enum: []string{"A", "B", "C", "D"}},
		},
	}
}
func (GPQADiamond) Validate(content string) (any, error) {
	var out GpqaOutput
	if err := json.Unmarshal([]byte(content), &out); err != nil {
		return nil, err
	}
	if !strings.ContainsRune("ABCD", rune(out.Answer[0])) || len(out.Answer) != 1 {
		return nil, fmt.Errorf("answer must be A/B/C/D, got %q", out.Answer)
	}
	return out, nil
}
func (GPQADiamond) Score(parsed any, gold json.RawMessage) (bool, map[string]any) {
	p, ok := parsed.(GpqaOutput)
	if !ok {
		return false, nil
	}
	var g string
	_ = json.Unmarshal(gold, &g)
	return strings.EqualFold(p.Answer, g), nil
}

// ── Budget hint helpers ───────────────────────────────────────────────────

func budgetHintMessage(remaining int) openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleSystem,
		Content: fmt.Sprintf(
			"[Budget: %d tokens remaining for this response. "+
				"Emit valid output before you run out — truncation = failure.]",
			remaining),
	}
}

func defaultOnFailure(comp *Completion, errMsg string) string {
	return fmt.Sprintf(
		"That response failed validation: %s. "+
			"Re-emit ONLY the expected output, no commentary, no markdown fences.",
		errMsg)
}

// ── Backend (OpenAI-compatible, with strict json_schema + json_object fallback) ──

type Backend struct {
	client    *openai.Client
	model     string
	mu        sync.Mutex
	strictOK  *bool // nil = unknown, &true = strict OK, &false = use json_object
}

func newBackend(baseURL, apiKey, model string) *Backend {
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = baseURL
	return &Backend{client: openai.NewClientWithConfig(cfg), model: model}
}

func (b *Backend) call(ctx context.Context, messages []openai.ChatCompletionMessage,
	bench Bench, maxTokens int, temperature, topP float32) (*Completion, error) {
	b.mu.Lock()
	tryStrict := b.strictOK == nil || *b.strictOK
	b.mu.Unlock()

	req := openai.ChatCompletionRequest{
		Model:       b.model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        topP,
	}

	if tryStrict {
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:   bench.Name() + "_output",
				Schema: bench.Schema(),
				Strict: true,
			},
		}
		resp, err := b.client.CreateChatCompletion(ctx, req)
		if err == nil {
			b.mu.Lock()
			t := true
			b.strictOK = &t
			b.mu.Unlock()
			return completionFrom(resp), nil
		}
		if isUnsupportedFormat(err) {
			b.mu.Lock()
			f := false
			b.strictOK = &f
			b.mu.Unlock()
		} else {
			return nil, err
		}
	}

	// json_object fallback
	req.ResponseFormat = &openai.ChatCompletionResponseFormat{
		Type: openai.ChatCompletionResponseFormatTypeJSONObject,
	}
	resp, err := b.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, err
	}
	return completionFrom(resp), nil
}

func isUnsupportedFormat(err error) bool {
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "response_format") || strings.Contains(s, "json_schema") ||
		strings.Contains(s, "unavailable") || strings.Contains(s, "not supported")
}

func completionFrom(resp openai.ChatCompletionResponse) *Completion {
	if len(resp.Choices) == 0 {
		return &Completion{}
	}
	c := resp.Choices[0]
	return &Completion{
		Content:          c.Message.Content,
		ReasoningContent: c.Message.ReasoningContent,
		FinishReason:     string(c.FinishReason),
		TokensUsed:       resp.Usage.CompletionTokens,
	}
}

// ── verify_loop: sequential call → validate → (on failure) feed back + retry ──

func verifyLoop(
	ctx context.Context,
	call func(ctx context.Context, msgs []openai.ChatCompletionMessage) (*Completion, error),
	validate func(content string) (any, error),
	initial []openai.ChatCompletionMessage,
	maxRetries int,
	budgetTokens *int, // nil = no budget hint
) *VerifyLoopOutcome {
	messages := append([]openai.ChatCompletionMessage{}, initial...)
	remaining := -1
	if budgetTokens != nil {
		remaining = *budgetTokens
	}
	out := &VerifyLoopOutcome{Messages: messages, History: []*Completion{}}

	for attempt := 0; attempt <= maxRetries; attempt++ {
		out.Attempts = attempt + 1

		if remaining >= 0 {
			messages = append(messages, budgetHintMessage(max(remaining, 0)))
		}

		comp, err := call(ctx, messages)
		out.Comp = comp
		if err != nil {
			out.FinalError = err.Error()
			out.Parsed = nil
		} else {
			out.History = append(out.History, comp)
			if remaining >= 0 {
				remaining -= comp.TokensUsed
			}
			parsed, vErr := validate(comp.Content)
			if vErr == nil {
				out.Parsed = parsed
				out.FinalError = ""
				out.Messages = messages
				return out
			}
			out.Parsed = nil
			out.FinalError = vErr.Error()
			if comp.FinishReason == "length" {
				out.Truncated = true
				if out.FinalError == "" {
					out.FinalError = "truncated"
				}
				out.Messages = messages
				return out
			}
		}

		badContent := ""
		if comp != nil {
			badContent = comp.Content
		}
		messages = append(messages,
			openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: badContent},
			openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: defaultOnFailure(comp, out.FinalError)},
		)
	}
	out.Messages = messages
	return out
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ── runOne + runBench ─────────────────────────────────────────────────────

func runOne(ctx context.Context, b *Backend, bench Bench, ex Example,
	maxTokens int, temperature, topP float32, maxRetries int, budgetHint bool) Result {

	t0 := time.Now()
	initial := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: bench.SystemPrompt()},
		{Role: openai.ChatMessageRoleUser, Content: ex.Prompt},
	}
	var budget *int
	if budgetHint {
		mt := maxTokens
		budget = &mt
	}

	outcome := verifyLoop(ctx,
		func(c context.Context, m []openai.ChatCompletionMessage) (*Completion, error) {
			return b.call(c, m, bench, maxTokens, temperature, topP)
		},
		bench.Validate, initial, maxRetries, budget,
	)
	elapsed := time.Since(t0).Seconds()

	r := Result{
		ID: ex.ID, Prompt: ex.Prompt, Gold: ex.Gold, Meta: ex.Meta, Elapsed: elapsed,
	}
	if outcome.Comp == nil {
		r.Error = outcome.FinalError
		r.Extras = map[string]any{"attempts": outcome.Attempts}
		return r
	}
	r.Response = outcome.Comp.Content
	r.Reasoning = outcome.Comp.ReasoningContent
	r.Parsed = outcome.Parsed

	correct, extras := false, map[string]any{}
	if outcome.Parsed != nil {
		correct, extras = bench.Score(outcome.Parsed, ex.Gold)
	}
	r.Correct = correct
	if extras == nil {
		extras = map[string]any{}
	}
	extras["finish"] = outcome.Comp.FinishReason
	extras["len_content"] = len(outcome.Comp.Content)
	extras["len_reasoning"] = len(outcome.Comp.ReasoningContent)
	extras["no_answer"] = outcome.Parsed == nil
	extras["attempts"] = outcome.Attempts
	extras["tokens_used"] = outcome.Comp.TokensUsed
	extras["truncated"] = outcome.Truncated
	extras["history"] = outcome.History
	extras["messages"] = outcome.Messages
	r.Extras = extras
	if outcome.FinalError != "" && outcome.Parsed == nil {
		r.Error = outcome.FinalError
	}
	return r
}

func runBench(ctx context.Context, b *Backend, bench Bench, examples []Example,
	concurrency int, dumpPath string, maxTokens int, temperature, topP float32,
	maxRetries int, budgetHint bool) {

	// Resume: skip examples whose IDs are already in the dump.
	done := loadDoneIDs(dumpPath)
	todo := []Example{}
	for _, ex := range examples {
		if !done[ex.ID] {
			todo = append(todo, ex)
		}
	}

	fmt.Printf("\n%s\n[%s] n=%d resumed=%d todo=%d concurrency=%d\n%s\n",
		strings.Repeat("=", 64), bench.Name(),
		len(examples), len(done), len(todo), concurrency,
		strings.Repeat("=", 64))

	dumpF, err := os.OpenFile(dumpPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatalf("open dump %s: %v", dumpPath, err)
	}
	defer dumpF.Close()

	results := make(chan Result, concurrency)
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	for _, ex := range todo {
		wg.Add(1)
		go func(ex Example) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			results <- runOne(ctx, b, bench, ex, maxTokens, temperature, topP, maxRetries, budgetHint)
		}(ex)
	}
	go func() { wg.Wait(); close(results) }()

	// Stream-dump every result + per-axis stats.
	stats := map[string]int{"correct": 0, "wrong": 0, "error": 0, "no_answer": 0, "total": 0}
	doneCount := len(done)
	for r := range results {
		doneCount++
		stats["total"]++
		mark := "-"
		switch {
		case r.Correct:
			stats["correct"]++
			mark = "+"
		case r.Error != "":
			stats["error"]++
			mark = "E"
		case noAns(r.Extras):
			stats["no_answer"]++
			mark = "?"
		default:
			stats["wrong"]++
		}
		line, _ := json.Marshal(r)
		dumpF.Write(line)
		dumpF.Write([]byte("\n"))
		fmt.Printf("  %s %3d/%d  id=%-12s  %5.1fs  fin=%-6s attempts=%d  tokens=%d\n",
			mark, doneCount, len(examples), r.ID, r.Elapsed,
			strOr(r.Extras["finish"], "?"), intOr(r.Extras["attempts"], 1),
			intOr(r.Extras["tokens_used"], 0))
	}

	if stats["total"] > 0 {
		fmt.Printf("\n[%s] correct=%d  wrong=%d  no_answer=%d  error=%d  (of %d fresh; +%d resumed)\n",
			bench.Name(), stats["correct"], stats["wrong"], stats["no_answer"], stats["error"],
			stats["total"], len(done))
	}
}

func loadDoneIDs(path string) map[string]bool {
	out := map[string]bool{}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	for {
		var r Result
		if err := dec.Decode(&r); err != nil {
			if err == io.EOF {
				break
			}
			continue
		}
		out[r.ID] = true
	}
	return out
}

func noAns(extras map[string]any) bool {
	v, ok := extras["no_answer"].(bool)
	return ok && v
}
func strOr(v any, def string) string {
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return def
}
func intOr(v any, def int) int {
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	}
	return def
}

// ── Frozen subset loader ──────────────────────────────────────────────────

func loadSubset(dir, name string, cap int) ([]Example, error) {
	path := filepath.Join(dir, name+".jsonl")
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open subset %s: %w", path, err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	out := []Example{}
	for {
		var ex Example
		if err := dec.Decode(&ex); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		out = append(out, ex)
		if cap > 0 && len(out) >= cap {
			break
		}
	}
	return out, nil
}

// ── main ──────────────────────────────────────────────────────────────────

func main() {
	model := flag.String("model", "vibethinker-3b-q4", "model id")
	baseURL := flag.String("base-url", "http://127.0.0.1:9100/v1", "OpenAI-compatible base URL")
	apiKey := flag.String("api-key", "dummy", "API key (oMLX ignores)")
	concurrency := flag.Int("concurrency", 2, "worker count")
	maxTokens := flag.Int("max-tokens", 40960, "per-call token budget")
	temperature := flag.Float64("temperature", 1.0, "sampling temperature")
	topP := flag.Float64("top-p", 0.95, "nucleus sampling cutoff")
	maxRetries := flag.Int("max-retries", 5, "verify-loop retries on parse fail")
	budgetHint := flag.Bool("budget-hint", false, "inject token-budget hint each call")
	n := flag.Int("n", 0, "cap per axis (0 = all)")
	benches := flag.String("benches", "zebralogic,cladder,gpqa_diamond", "comma list")
	subsetsDir := flag.String("subsets-dir", "../data/bench_subsets", "frozen subset dir")
	dumpDir := flag.String("dump-dir", "../data/bench_runs/go-bench", "output dir")
	flag.Parse()

	if err := os.MkdirAll(*dumpDir, 0o755); err != nil {
		log.Fatalf("mkdir dump: %v", err)
	}

	registry := map[string]Bench{
		"zebralogic":   ZebraLogic{},
		"cladder":      CLadder{},
		"gpqa_diamond": GPQADiamond{},
	}

	backend := newBackend(*baseURL, *apiKey, *model)
	ctx := context.Background()

	for _, name := range strings.Split(*benches, ",") {
		bench, ok := registry[strings.TrimSpace(name)]
		if !ok {
			log.Printf("unknown bench: %s (valid: zebralogic,cladder,gpqa_diamond)", name)
			continue
		}
		exs, err := loadSubset(*subsetsDir, bench.Name(), *n)
		if err != nil {
			log.Fatalf("load %s: %v", bench.Name(), err)
		}
		dumpPath := filepath.Join(*dumpDir, fmt.Sprintf("%s__%s.jsonl",
			strings.ReplaceAll(*model, "/", "_"), bench.Name()))
		runBench(ctx, backend, bench, exs, *concurrency, dumpPath,
			*maxTokens, float32(*temperature), float32(*topP), *maxRetries, *budgetHint)
	}
}
