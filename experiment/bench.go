// Reasoning + decision-under-uncertainty + knowledge bench. Bench
// definitions + orchestration only — the chat-completion loop and OpenAI
// backend live in harness.go and backend.go.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"math/rand"
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

// BenchSpec defines a bench. NewStep is called per example and returns
// (a) the user-role prompt to start the conversation and (b) a stateful
// step closure: given the assistant's latest content, either return a
// final T (isFinal=true) or an intermediate hint string to feed back as
// a system message (isFinal=false). Same shape handles single-shot
// validation (parse fail → intermediate hint, parse OK → final) and
// rollouts (env step → intermediate observation, env done → final).
type BenchSpec[T any] struct {
	Name         string
	SystemPrompt string
	Schema       *jsonschema.Definition
	NewStep      func(ex Example[T]) (userPrompt string, step func(content string) (final T, hint string, isFinal bool))
	Score        func(parsed, gold T) (correct bool, extras map[string]any)
}

type Result[T any] struct {
	ID       string                          `json:"id"`
	Prompt   string                          `json:"prompt"`
	Gold     T                               `json:"gold"`
	Parsed   *T                              `json:"parsed"`
	Correct  bool                            `json:"correct"`
	Elapsed  float64                         `json:"elapsed"`
	Error    string                          `json:"error"`
	Meta     map[string]any                  `json:"meta"`
	Extras   map[string]any                  `json:"extras"`
	History  []openai.ChatCompletionResponse `json:"history"`
	Messages []openai.ChatCompletionMessage  `json:"messages"`
}

// ── Helpers ───────────────────────────────────────────────────────────────

// extractLastJSON returns the last balanced {...} substring in s, or s
// unchanged if none is found. Used in tolerant mode for non-thinking
// models that wrap output in prose/markdown fences.
func extractLastJSON(s string) string {
	end := -1
	depth := 0
	for i := len(s) - 1; i >= 0; i-- {
		switch s[i] {
		case '}':
			if depth == 0 {
				end = i
			}
			depth++
		case '{':
			depth--
			if depth == 0 && end >= 0 {
				return s[i : end+1]
			}
		}
	}
	return s
}

// qwen3Assistant folds reasoning_content back into the assistant message
// using <think>...</think> tags so subsequent turns see the prior CoT.
func qwen3Assistant(r openai.ChatCompletionResponse) openai.ChatCompletionMessage {
	msg := r.Choices[0].Message
	content := msg.Content
	if msg.ReasoningContent != "" {
		content = "<think>\n" + msg.ReasoningContent + "\n</think>\n" + content
	}
	return openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: content,
	}
}

// defaultAssistant preserves ReasoningContent as a separate field instead
// of embedding it into Content. Suitable for models whose tokenizer chat
// template handles reasoning independently (Gemma, etc.).
func defaultAssistant(r openai.ChatCompletionResponse) openai.ChatCompletionMessage {
	msg := r.Choices[0].Message
	return openai.ChatCompletionMessage{
		Role:             openai.ChatMessageRoleAssistant,
		Content:          msg.Content,
		ReasoningContent: msg.ReasoningContent,
	}
}

func assistantFnFor(fmtName string) func(openai.ChatCompletionResponse) openai.ChatCompletionMessage {
	switch fmtName {
	case "qwen3":
		return qwen3Assistant
	case "gemma4", "default":
		return defaultAssistant
	default:
		log.Fatalf("unknown --assistant-fmt %q (want qwen3 | gemma4 | default)", fmtName)
		return nil
	}
}

// ── ZebraLogic ────────────────────────────────────────────────────────────

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
			Type:                 jsonschema.Object,
			Required:             []string{"header", "rows"},
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"header": {Type: jsonschema.Array, Items: &jsonschema.Definition{Type: jsonschema.String}},
				"rows":   {Type: jsonschema.Array, Items: &jsonschema.Definition{Type: jsonschema.Array, Items: &jsonschema.Definition{Type: jsonschema.String}}},
			},
		},
		NewStep: func(ex Example[ZebraOutput]) (string, func(string) (ZebraOutput, string, bool)) {
			step := func(c string) (ZebraOutput, string, bool) {
				var z ZebraOutput
				if e := json.Unmarshal([]byte(c), &z); e != nil {
					hint := fmt.Sprintf("invalid JSON: %v.\n"+
						`Expected exactly: {"header": ["House","Name","Color",...], "rows": [["1","Alice","red",...], ["2","Bob","blue",...]]}.`+"\n"+
						"Each row MUST be its own inner [...] array nested inside the outer rows array.", e)
					return z, hint, false
				}
				if len(z.Header) == 0 || len(z.Rows) == 0 {
					return z, "header and rows must both be non-empty arrays", false
				}
				for i, r := range z.Rows {
					if len(r) != len(z.Header) {
						return z, fmt.Sprintf("row %d has %d cells but header has %d", i, len(r), len(z.Header)), false
					}
				}
				return z, "", true
			}
			return ex.Prompt, step
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

// ── GPQA-Diamond ──────────────────────────────────────────────────────────

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
			Type:                 jsonschema.Object,
			Required:             []string{"answer"},
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"answer": {Type: jsonschema.String, Enum: []string{"A", "B", "C", "D"}},
			},
		},
		NewStep: func(ex Example[GpqaOutput]) (string, func(string) (GpqaOutput, string, bool)) {
			step := func(c string) (GpqaOutput, string, bool) {
				var o GpqaOutput
				if e := json.Unmarshal([]byte(c), &o); e != nil {
					return o, fmt.Sprintf(`invalid JSON: %v. Expected exactly: {"answer": "A"} (or B, C, D)`, e), false
				}
				if len(o.Answer) != 1 || !strings.ContainsRune("ABCD", rune(o.Answer[0])) {
					return o, fmt.Sprintf(`answer must be a single letter A, B, C, or D (got %q)`, o.Answer), false
				}
				return o, "", true
			}
			return ex.Prompt, step
		},
		Score: func(p, g GpqaOutput) (bool, map[string]any) {
			return strings.EqualFold(p.Answer, g.Answer), nil
		},
	}
}

// ── K-armed Bernoulli bandit (decision under uncertainty) ─────────────────

// BanditRow's Gold = {K, Probs}; the rollout trace is {Actions, Rewards}.
// T (rounds per game) is a spec-level constant. Rewards are sampled at
// step time from a per-example seeded RNG so runs are reproducible
// across models without storing a reward table in the data.
type BanditRow struct {
	K       int       `json:"K"`
	Probs   []float64 `json:"probs"`
	Actions []int     `json:"actions,omitempty"`
	Rewards []int     `json:"rewards,omitempty"`
}

const banditT = 30 // rounds per game

func banditBench() BenchSpec[BanditRow] {
	return BenchSpec[BanditRow]{
		Name: "bandit",
		SystemPrompt: "You play a multi-armed bandit game. Each arm has a fixed but " +
			"unknown success probability. On every round you pick one arm and observe " +
			"a Bernoulli reward (0 or 1). Maximize total reward over the full game.",
		Schema: &jsonschema.Definition{
			Type:                 jsonschema.Object,
			Required:             []string{"action"},
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"action": {Type: jsonschema.Integer},
			},
		},
		NewStep: func(ex Example[BanditRow]) (string, func(string) (BanditRow, string, bool)) {
			env := ex.Gold
			state := BanditRow{K: env.K, Probs: env.Probs}
			// Per-example deterministic RNG seeded by ID hash → identical
			// reward streams across model runs without storing a table.
			h := fnv.New64a()
			h.Write([]byte(ex.ID))
			envRng := rand.New(rand.NewSource(int64(h.Sum64())))

			userPrompt := fmt.Sprintf(
				"Game: %d arms (numbered 1..%d), %d rounds total. Each arm's success "+
					"probability is fixed for the whole game but unknown to you. "+
					"Reward each round is 0 or 1.\n\n"+
					"Respond each round with a single JSON object: {\"action\": <int 1..%d>}.\n\n"+
					"Round 1 / %d. Which arm?", env.K, env.K, banditT, env.K, banditT)

			step := func(c string) (BanditRow, string, bool) {
				var act struct {
					Action int `json:"action"`
				}
				if e := json.Unmarshal([]byte(c), &act); e != nil {
					return state, fmt.Sprintf(`invalid JSON: %v. Expected: {"action": <int 1..%d>}`, e, env.K), false
				}
				if act.Action < 1 || act.Action > env.K {
					return state, fmt.Sprintf("action must be in 1..%d (got %d)", env.K, act.Action), false
				}
				round := len(state.Actions) // 0-indexed round we're resolving
				reward := 0
				if envRng.Float64() < env.Probs[act.Action-1] {
					reward = 1
				}
				state.Actions = append(state.Actions, act.Action)
				state.Rewards = append(state.Rewards, reward)
				if len(state.Actions) >= banditT {
					return state, "", true
				}
				next := len(state.Actions) + 1
				return state, fmt.Sprintf(
					"Round %d result: arm %d → reward %d.\n"+
						"Round %d / %d. Which arm?",
					round+1, act.Action, reward, next, banditT), false
			}
			return userPrompt, step
		},
		Score: func(p, g BanditRow) (bool, map[string]any) {
			optimal := 0.0
			meanArm := 0.0
			for _, pr := range g.Probs {
				if pr > optimal {
					optimal = pr
				}
				meanArm += pr
			}
			meanArm /= float64(len(g.Probs))
			uniformInstantRegret := optimal - meanArm

			// Sum instantaneous regret over all banditT rounds. Rounds
			// played count their actual regret; rounds skipped (budget /
			// step cap exhausted) are filled with the uniform-random
			// regret, so truncation is penalized.
			totalRegret := 0.0
			for _, a := range p.Actions {
				totalRegret += optimal - g.Probs[a-1]
			}
			unplayed := banditT - len(p.Actions)
			if unplayed < 0 {
				unplayed = 0
			}
			totalRegret += float64(unplayed) * uniformInstantRegret
			meanRegret := totalRegret / float64(banditT)

			totalReward := 0
			for _, r := range p.Rewards {
				totalReward += r
			}
			return meanRegret < uniformInstantRegret, map[string]any{
				"mean_regret":    meanRegret,
				"uniform_regret": uniformInstantRegret,
				"optimal_p":      optimal,
				"total_reward":   totalReward,
				"rounds_played":  len(p.Actions),
				"rounds_skipped": unplayed,
			}
		},
	}
}

// ── runBench + runOne ─────────────────────────────────────────────────────

func runBench[T any](ctx context.Context, b Backend, spec BenchSpec[T],
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
		fmt.Printf("  %s %3d/%d  id=%-14s  %5.1fs  steps=%d  tokens=%d\n",
			mark, doneCount, len(examples), r.ID, r.Elapsed,
			intOr(r.Extras["steps"], 1), intOr(r.Extras["tokens_used"], 0))
	}

	if stats["total"] > 0 {
		fmt.Printf("\n[%s] correct=%d wrong=%d no_answer=%d error=%d  (of %d fresh; +%d resumed)\n",
			spec.Name, stats["correct"], stats["wrong"], stats["no_answer"], stats["error"],
			stats["total"], len(done))
	}
}

func runOne[T any](ctx context.Context, b Backend, spec BenchSpec[T],
	ex Example[T], opts Opts) Result[T] {

	t0 := time.Now()
	userPrompt, step := spec.NewStep(ex)
	if opts.TolerantJSON {
		inner := step
		step = func(c string) (T, string, bool) { return inner(extractLastJSON(c)) }
	}
	sysPrompt := spec.SystemPrompt
	if opts.SystemSuffix != "" {
		sysPrompt = sysPrompt + "\n\n" + opts.SystemSuffix
	}
	initial := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: sysPrompt},
		{Role: openai.ChatMessageRoleUser, Content: userPrompt},
	}
	completion := func(c context.Context, m []openai.ChatCompletionMessage, maxTok int) (*openai.ChatCompletionResponse, error) {
		return b.CallWithSchema(c, m, spec.Schema, spec.Name+"_output",
			maxTok, opts.Temperature, opts.TopP)
	}
	assistantFn := opts.AssistantFn
	if assistantFn == nil {
		assistantFn = qwen3Assistant
	}
	parsed, history, err := ChatCompletionLoop(ctx, initial, completion,
		assistantFn, step, opts.MaxSteps, opts.MaxTokens)

	r := Result[T]{
		ID: ex.ID, Prompt: userPrompt, Gold: ex.Gold, Meta: ex.Meta,
		Elapsed:  time.Since(t0).Seconds(),
		History:  history.Responses, Messages: history.Messages,
	}
	extras := map[string]any{
		"steps":       len(history.Responses),
		"exit_reason": string(history.Reason),
		"mode":        b.Mode(),
	}
	if n := len(history.Responses); n > 0 {
		total := 0
		for _, h := range history.Responses {
			total += h.Usage.CompletionTokens
		}
		extras["tokens_used"] = total
		last := history.Responses[n-1]
		if len(last.Choices) > 0 {
			extras["finish"] = string(last.Choices[0].FinishReason)
		}
	}
	if err == nil && history.Reason == ExitOK {
		r.Parsed = &parsed
		ok, sx := spec.Score(parsed, ex.Gold)
		r.Correct = ok
		for k, v := range sx {
			extras[k] = v
		}
	} else if err != nil {
		r.Error = err.Error()
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
	Concurrency, MaxTokens, MaxSteps int
	Temperature, TopP                float32
	TolerantJSON                     bool
	SystemSuffix                     string
	AssistantFn                      func(openai.ChatCompletionResponse) openai.ChatCompletionMessage
}

func runNamed[T any](ctx context.Context, b Backend, spec BenchSpec[T],
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
	maxSteps := flag.Int("max-steps", 30, "loop safety cap (env steps + retries)")
	tolerantJSON := flag.Bool("tolerant-json", false, "extract last {...} block from content (for non-thinking models that wrap output)")
	n := flag.Int("n", 0, "cap per axis (0 = all)")
	benches := flag.String("benches", "zebralogic,bandit,gpqa_diamond", "comma list")
	subsetsDir := flag.String("subsets-dir", "../data/bench_subsets", "frozen subset dir")
	dumpDir := flag.String("dump-dir", "../data/bench_runs/go-bench", "output dir")
	backendKind := flag.String("backend", "openai", "openai | claude-cli")
	cliEffort := flag.String("cli-effort", "medium", "claude -p --effort (low|medium|high|xhigh|max)")
	assistantFmt := flag.String("assistant-fmt", "qwen3", "response → assistant message conversion: qwen3 (wrap reasoning in <think>) | gemma4 | default (pass ReasoningContent through)")
	sysSuffix := flag.String("system-suffix", "", "appended to each bench's SystemPrompt (use for non-Qwen3 family thinking-off conventions; use --thinking-off instead for Qwen3 family)")
	thinkingOff := flag.Bool("thinking-off", false, "send chat_template_kwargs={\"enable_thinking\": false} (Qwen3 family — works at the chat template, not as a prompt suffix)")
	dumpTag := flag.String("dump-tag", "", "appended to model id in the JSONL filename (use to keep configs of the same model in separate files, e.g. '-think' vs '-nothink')")
	flag.Parse()

	if err := os.MkdirAll(*dumpDir, 0o755); err != nil {
		log.Fatalf("mkdir dump: %v", err)
	}
	opts := Opts{
		Concurrency: *concurrency, MaxTokens: *maxTokens, MaxSteps: *maxSteps,
		Temperature: float32(*temperature), TopP: float32(*topP),
		TolerantJSON: *tolerantJSON, AssistantFn: assistantFnFor(*assistantFmt),
		SystemSuffix: *sysSuffix,
	}
	var tmplKwargs map[string]any
	if *thinkingOff {
		tmplKwargs = map[string]any{"enable_thinking": false}
	}
	var backend Backend
	switch *backendKind {
	case "claude-cli":
		backend = NewClaudeCLIBackend(*model, *cliEffort)
	default:
		backend = NewOpenAIBackend(*baseURL, *apiKey, *model, tmplKwargs)
	}
	ctx := context.Background()

	dumpFor := func(name string) string {
		return filepath.Join(*dumpDir, fmt.Sprintf("%s%s__%s.jsonl",
			strings.ReplaceAll(*model, "/", "_"), *dumpTag, name))
	}

	for _, name := range strings.Split(*benches, ",") {
		name = strings.TrimSpace(name)
		switch name {
		case "zebralogic":
			runNamed(ctx, backend, zebraBench(), *subsetsDir, dumpFor(name), *n, opts)
		case "bandit":
			runNamed(ctx, backend, banditBench(), *subsetsDir, dumpFor(name), *n, opts)
		case "gpqa_diamond":
			runNamed(ctx, backend, gpqaBench(), *subsetsDir, dumpFor(name), *n, opts)
		default:
			log.Printf("unknown bench: %s", name)
		}
	}
}
