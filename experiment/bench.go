// Three-axis reasoning bench. Bench definitions + orchestration only —
// the chat-completion loop and OpenAI backend live in harness.go.
//
//   go run . --model vibethinker-3b-q4 --concurrency 2 --n 10 \
//            --max-retries 5 --top-p 0.95 --max-tokens 40960 \
//            --dump-dir ../data/bench_runs/test
package main

import (
	"context"
	"encoding/json"
	"errors"
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

// ── Types ─────────────────────────────────────────────────────────────────

type Example[T any] struct {
	ID     string         `json:"id"`
	Prompt string         `json:"prompt"`
	Gold   T              `json:"gold"`
	Meta   map[string]any `json:"meta"`
}

type BenchSpec[T any] struct {
	Name         string
	SystemPrompt string
	Schema       *jsonschema.Definition
	Validate     func(content string) (T, error)
	Score        func(parsed, gold T) (correct bool, extras map[string]any)
}

type Result[T any] struct {
	ID       string                           `json:"id"`
	Prompt   string                           `json:"prompt"`
	Gold     T                                `json:"gold"`
	Parsed   *T                               `json:"parsed"`
	Correct  bool                             `json:"correct"`
	Elapsed  float64                          `json:"elapsed"`
	Error    string                           `json:"error"`
	Meta     map[string]any                   `json:"meta"`
	Extras   map[string]any                   `json:"extras"`
	History  []openai.ChatCompletionResponse  `json:"history"`
	Messages []openai.ChatCompletionMessage   `json:"messages"`
}

// ── Bench specs ───────────────────────────────────────────────────────────

type ZebraOutput struct {
	Header []string   `json:"header"`
	Rows   [][]string `json:"rows"`
}

func zebraBench() BenchSpec[ZebraOutput] {
	return BenchSpec[ZebraOutput]{
		Name: "zebralogic",
		SystemPrompt: "You solve logic-grid puzzles. Respond with a single JSON " +
			"object matching this schema (no markdown, no commentary, just JSON):\n" +
			`  {"header": ["House", "<attr1>", ...], "rows": [["1", "...", ...], ...]}` + "\n" +
			"Use the exact attribute values from the puzzle.",
		Schema: &jsonschema.Definition{
			Type:     jsonschema.Object,
			Required: []string{"header", "rows"},
			Properties: map[string]jsonschema.Definition{
				"header": {Type: jsonschema.Array, Items: &jsonschema.Definition{Type: jsonschema.String}},
				"rows":   {Type: jsonschema.Array, Items: &jsonschema.Definition{Type: jsonschema.Array, Items: &jsonschema.Definition{Type: jsonschema.String}}},
			},
		},
		Validate: func(c string) (ZebraOutput, error) {
			var z ZebraOutput
			return z, json.Unmarshal([]byte(c), &z)
		},
		Score: func(p, g ZebraOutput) (bool, map[string]any) {
			total, correct := 0, 0
			for i, gr := range g.Rows {
				total += len(gr)
				if i >= len(p.Rows) {
					continue
				}
				for j, gc := range gr {
					if j < len(p.Rows[i]) && strings.EqualFold(strings.TrimSpace(gc), strings.TrimSpace(p.Rows[i][j])) {
						correct++
					}
				}
			}
			return correct == total && len(g.Rows) == len(p.Rows),
				map[string]any{"cell_correct": correct, "cell_total": total}
		},
	}
}

// CladderOutput / GpqaOutput accept either a bare string (frozen-subset
// gold format) or {"answer": ...} (validation format).

type CladderOutput struct {
	Answer string `json:"answer"`
}

func (c *CladderOutput) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		c.Answer = s
		return nil
	}
	type alias CladderOutput
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*c = CladderOutput(a)
	return nil
}

func cladderBench() BenchSpec[CladderOutput] {
	return BenchSpec[CladderOutput]{
		Name: "cladder",
		SystemPrompt: "Answer the causal-reasoning question. Respond with a single " +
			`JSON object: {"answer": "yes"} or {"answer": "no"}.`,
		Schema: &jsonschema.Definition{
			Type:     jsonschema.Object,
			Required: []string{"answer"},
			Properties: map[string]jsonschema.Definition{
				"answer": {Type: jsonschema.String, Enum: []string{"yes", "no"}},
			},
		},
		Validate: func(c string) (CladderOutput, error) {
			var o CladderOutput
			if err := json.Unmarshal([]byte(c), &o); err != nil {
				return o, err
			}
			if o.Answer != "yes" && o.Answer != "no" {
				return o, fmt.Errorf("answer must be yes/no, got %q", o.Answer)
			}
			return o, nil
		},
		Score: func(p, g CladderOutput) (bool, map[string]any) {
			return strings.EqualFold(strings.TrimSpace(p.Answer), strings.TrimSpace(g.Answer)), nil
		},
	}
}

type GpqaOutput struct {
	Answer string `json:"answer"`
}

func (g *GpqaOutput) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		g.Answer = s
		return nil
	}
	type alias GpqaOutput
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*g = GpqaOutput(a)
	return nil
}

func gpqaBench() BenchSpec[GpqaOutput] {
	return BenchSpec[GpqaOutput]{
		Name: "gpqa_diamond",
		SystemPrompt: "Answer the multiple-choice question. Respond with a single " +
			`JSON object: {"answer": "A"} or "B", "C", "D".`,
		Schema: &jsonschema.Definition{
			Type:     jsonschema.Object,
			Required: []string{"answer"},
			Properties: map[string]jsonschema.Definition{
				"answer": {Type: jsonschema.String, Enum: []string{"A", "B", "C", "D"}},
			},
		},
		Validate: func(c string) (GpqaOutput, error) {
			var o GpqaOutput
			if err := json.Unmarshal([]byte(c), &o); err != nil {
				return o, err
			}
			if len(o.Answer) != 1 || !strings.ContainsRune("ABCD", rune(o.Answer[0])) {
				return o, fmt.Errorf("answer must be A/B/C/D, got %q", o.Answer)
			}
			return o, nil
		},
		Score: func(p, g GpqaOutput) (bool, map[string]any) {
			return strings.EqualFold(p.Answer, g.Answer), nil
		},
	}
}

// ── runBench + runOne ─────────────────────────────────────────────────────

func runBench[T any](ctx context.Context, b *Backend, spec BenchSpec[T],
	examples []Example[T], dumpPath string, opts Opts) {

	done := loadDoneIDs(dumpPath)
	todo := []Example[T]{}
	for _, ex := range examples {
		if !done[ex.ID] {
			todo = append(todo, ex)
		}
	}
	fmt.Printf("\n%s\n[%s] n=%d resumed=%d todo=%d concurrency=%d\n%s\n",
		strings.Repeat("=", 64), spec.Name, len(examples), len(done), len(todo),
		opts.Concurrency, strings.Repeat("=", 64))

	dumpF, err := os.OpenFile(dumpPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatalf("open dump %s: %v", dumpPath, err)
	}
	defer dumpF.Close()

	results := make(chan Result[T], opts.Concurrency)
	sem := make(chan struct{}, opts.Concurrency)
	var wg sync.WaitGroup
	for _, ex := range todo {
		wg.Add(1)
		go func(ex Example[T]) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			results <- runOne(ctx, b, spec, ex, opts)
		}(ex)
	}
	go func() { wg.Wait(); close(results) }()

	stats := map[string]int{}
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
		case r.Parsed == nil:
			stats["no_answer"]++
			mark = "?"
		default:
			stats["wrong"]++
		}
		line, _ := json.Marshal(r)
		dumpF.Write(line)
		dumpF.Write([]byte("\n"))
		fmt.Printf("  %s %3d/%d  id=%-14s  %5.1fs  attempts=%d  tokens=%d\n",
			mark, doneCount, len(examples), r.ID, r.Elapsed,
			intOr(r.Extras["attempts"], 1), intOr(r.Extras["tokens_used"], 0))
	}

	if stats["total"] > 0 {
		fmt.Printf("\n[%s] correct=%d wrong=%d no_answer=%d error=%d  (of %d fresh; +%d resumed)\n",
			spec.Name, stats["correct"], stats["wrong"], stats["no_answer"], stats["error"],
			stats["total"], len(done))
	}
}

func runOne[T any](ctx context.Context, b *Backend, spec BenchSpec[T],
	ex Example[T], opts Opts) Result[T] {

	t0 := time.Now()
	initial := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: spec.SystemPrompt},
		{Role: openai.ChatMessageRoleUser, Content: ex.Prompt},
	}
	completion := func(c context.Context, m []openai.ChatCompletionMessage, maxTok int) (*openai.ChatCompletionResponse, error) {
		return b.CallWithSchema(c, m, spec.Schema, spec.Name+"_output",
			maxTok, opts.Temperature, opts.TopP)
	}
	parsed, history, err := ChatCompletionLoop(ctx, initial, completion,
		spec.Validate, opts.MaxRetries, opts.MaxTokens)

	r := Result[T]{
		ID: ex.ID, Prompt: ex.Prompt, Gold: ex.Gold, Meta: ex.Meta,
		Elapsed:  time.Since(t0).Seconds(),
		History:  history.Responses, Messages: history.Messages,
	}
	extras := map[string]any{"attempts": len(history.Responses)}
	if n := len(history.Responses); n > 0 {
		last := history.Responses[n-1]
		extras["tokens_used"] = last.Usage.CompletionTokens
		if len(last.Choices) > 0 {
			extras["finish"] = string(last.Choices[0].FinishReason)
		}
	}
	if err == nil {
		r.Parsed = &parsed
		ok, sx := spec.Score(parsed, ex.Gold)
		r.Correct = ok
		for k, v := range sx {
			extras[k] = v
		}
	} else {
		r.Error = err.Error()
		var le *LoopError
		if errors.As(err, &le) {
			extras["exit_reason"] = string(le.Reason)
		}
	}
	r.Extras = extras
	return r
}

// ── Helpers ───────────────────────────────────────────────────────────────

func loadDoneIDs(path string) map[string]bool {
	out := map[string]bool{}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	for {
		var r struct {
			ID string `json:"id"`
		}
		if err := dec.Decode(&r); err == io.EOF {
			break
		} else if err == nil {
			out[r.ID] = true
		}
	}
	return out
}

func loadSubset[T any](dir, name string, n int) ([]Example[T], error) {
	f, err := os.Open(filepath.Join(dir, name+".jsonl"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	out := []Example[T]{}
	for {
		var ex Example[T]
		if err := dec.Decode(&ex); err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		out = append(out, ex)
		if n > 0 && len(out) >= n {
			break
		}
	}
	return out, nil
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

// ── main ──────────────────────────────────────────────────────────────────

type Opts struct {
	Concurrency, MaxTokens, MaxRetries int
	Temperature, TopP                  float32
}

func runNamed[T any](ctx context.Context, b *Backend, spec BenchSpec[T],
	subsetsDir, dumpPath string, n int, opts Opts) {
	exs, err := loadSubset[T](subsetsDir, spec.Name, n)
	if err != nil {
		log.Fatalf("load %s: %v", spec.Name, err)
	}
	runBench(ctx, b, spec, exs, dumpPath, opts)
}

func main() {
	model := flag.String("model", "vibethinker-3b-q4", "model id")
	baseURL := flag.String("base-url", "http://127.0.0.1:9100/v1", "OpenAI-compatible base URL")
	apiKey := flag.String("api-key", "dummy", "API key")
	concurrency := flag.Int("concurrency", 2, "worker count")
	maxTokens := flag.Int("max-tokens", 40960, "per-call token budget")
	temperature := flag.Float64("temperature", 1.0, "sampling temperature")
	topP := flag.Float64("top-p", 0.95, "nucleus sampling cutoff")
	maxRetries := flag.Int("max-retries", 5, "verify-loop retries on parse fail")
	n := flag.Int("n", 0, "cap per axis (0 = all)")
	benches := flag.String("benches", "zebralogic,cladder,gpqa_diamond", "comma list")
	subsetsDir := flag.String("subsets-dir", "../data/bench_subsets", "frozen subset dir")
	dumpDir := flag.String("dump-dir", "../data/bench_runs/go-bench", "output dir")
	flag.Parse()

	if err := os.MkdirAll(*dumpDir, 0o755); err != nil {
		log.Fatalf("mkdir dump: %v", err)
	}
	opts := Opts{
		Concurrency: *concurrency, MaxTokens: *maxTokens, MaxRetries: *maxRetries,
		Temperature: float32(*temperature), TopP: float32(*topP),
	}
	backend := NewBackend(*baseURL, *apiKey, *model)
	ctx := context.Background()

	dumpFor := func(name string) string {
		return filepath.Join(*dumpDir, fmt.Sprintf("%s__%s.jsonl",
			strings.ReplaceAll(*model, "/", "_"), name))
	}

	for _, name := range strings.Split(*benches, ",") {
		name = strings.TrimSpace(name)
		switch name {
		case "zebralogic":
			runNamed(ctx, backend, zebraBench(), *subsetsDir, dumpFor(name), *n, opts)
		case "cladder":
			runNamed(ctx, backend, cladderBench(), *subsetsDir, dumpFor(name), *n, opts)
		case "gpqa_diamond":
			runNamed(ctx, backend, gpqaBench(), *subsetsDir, dumpFor(name), *n, opts)
		default:
			log.Printf("unknown bench: %s", name)
		}
	}
}
