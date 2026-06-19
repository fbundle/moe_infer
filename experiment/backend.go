package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"sync"

	openai "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

// Backend is the contract every contender (OpenAI-compatible HTTP, claude CLI, ...) implements.
type Backend interface {
	CallWithSchema(ctx context.Context,
		messages []openai.ChatCompletionMessage,
		schema *jsonschema.Definition, schemaName string,
		maxTokens int, temperature, topP float32,
	) (*openai.ChatCompletionResponse, error)
	Mode() string
}

// ── OpenAI-compatible backend (strict json_schema → json_object fallback) ──

type OpenAIBackend struct {
	client       *openai.Client
	model        string
	tmplKwargs   map[string]any // sent as chat_template_kwargs (e.g. {"enable_thinking": false})
	mu           sync.Mutex
	strictOK     *bool // nil = unknown
}

func NewOpenAIBackend(baseURL, apiKey, model string, tmplKwargs map[string]any) *OpenAIBackend {
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = baseURL
	return &OpenAIBackend{client: openai.NewClientWithConfig(cfg), model: model, tmplKwargs: tmplKwargs}
}

func (b *OpenAIBackend) CallWithSchema(
	ctx context.Context,
	messages []openai.ChatCompletionMessage,
	schema *jsonschema.Definition, schemaName string,
	maxTokens int, temperature, topP float32,
) (*openai.ChatCompletionResponse, error) {
	b.mu.Lock()
	tryStrict := b.strictOK == nil || *b.strictOK
	b.mu.Unlock()

	req := openai.ChatCompletionRequest{
		Model: b.model, Messages: messages, MaxTokens: maxTokens,
		Temperature: temperature, TopP: topP,
		ChatTemplateKwargs: b.tmplKwargs,
	}
	if tryStrict {
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name: schemaName, Schema: schema, Strict: true,
			},
		}
		resp, err := b.client.CreateChatCompletion(ctx, req)
		if err == nil {
			b.setStrict(true)
			return &resp, nil
		}
		s := strings.ToLower(err.Error())
		if strings.Contains(s, "response_format") || strings.Contains(s, "json_schema") ||
			strings.Contains(s, "unavailable") || strings.Contains(s, "not supported") {
			b.setStrict(false)
		} else {
			return nil, err
		}
	}
	req.ResponseFormat = &openai.ChatCompletionResponseFormat{
		Type: openai.ChatCompletionResponseFormatTypeJSONObject,
	}
	resp, err := b.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (b *OpenAIBackend) setStrict(v bool) {
	b.mu.Lock()
	b.strictOK = &v
	b.mu.Unlock()
}

func (b *OpenAIBackend) Mode() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.strictOK == nil {
		return "unknown"
	}
	if *b.strictOK {
		return "strict"
	}
	return "json_object"
}

// ── Claude Code headless CLI backend (`claude -p`) ────────────────────────
//
// One fresh CLI process per call. Multi-turn message history is flattened
// into a single prompt with ROLE: labels — fine for the verify-loop case
// where we replay <prev assistant>+<retry hint>. `--system-prompt` fully
// replaces the default system prompt, suppressing CLAUDE.md auto-discovery
// and auto-memory so the contender sees the same inputs as a blank API call.

type ClaudeCLIBackend struct {
	model  string
	effort string
}

func NewClaudeCLIBackend(model, effort string) *ClaudeCLIBackend {
	return &ClaudeCLIBackend{model: model, effort: effort}
}

type cliEnvelope struct {
	Result           string          `json:"result"`
	StructuredOutput json.RawMessage `json:"structured_output"`
	SessionID        string          `json:"session_id"`
	TotalCost        float64         `json:"total_cost_usd"`
	IsError          bool            `json:"is_error"`
	StopReason       string          `json:"stop_reason"`
	Usage            struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func strOr(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

func (b *ClaudeCLIBackend) CallWithSchema(
	ctx context.Context,
	messages []openai.ChatCompletionMessage,
	schema *jsonschema.Definition, schemaName string,
	_ int, _, _ float32,
) (*openai.ChatCompletionResponse, error) {
	var sys strings.Builder
	var convo []string
	for _, m := range messages {
		switch m.Role {
		case openai.ChatMessageRoleSystem:
			if sys.Len() > 0 {
				sys.WriteString("\n\n")
			}
			sys.WriteString(m.Content)
		case openai.ChatMessageRoleUser:
			convo = append(convo, "USER:\n"+m.Content)
		case openai.ChatMessageRoleAssistant:
			convo = append(convo, "ASSISTANT:\n"+m.Content)
		}
	}
	prompt := strings.Join(convo, "\n\n")

	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("marshal schema: %w", err)
	}

	args := []string{
		"-p",
		"--model", b.model,
		"--effort", b.effort,
		"--system-prompt", sys.String(),
		"--json-schema", string(schemaJSON),
		"--output-format", "json",
		"--no-session-persistence",
		"--dangerously-skip-permissions",
		prompt,
	}

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Stdin = bytes.NewReader(nil)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	// Prefer the structured envelope when present, even on non-zero exit:
	// `claude -p` returns exit 1 with a populated JSON envelope on policy
	// refusals (stop_reason="refusal", is_error=true) and on other in-band
	// errors. The exit code alone hides the real reason.
	var env cliEnvelope
	if json.Unmarshal(stdout.Bytes(), &env) == nil {
		if env.IsError || env.StopReason == "refusal" {
			return nil, fmt.Errorf("claude -p %s: %s",
				strOr(env.StopReason, "is_error"), env.Result)
		}
		if runErr != nil {
			return nil, fmt.Errorf("claude -p failed: %v (envelope: %q)",
				runErr, env.Result)
		}
	} else if runErr != nil {
		errOut := stderr.String()
		if len(errOut) > 400 {
			errOut = errOut[:400]
		}
		return nil, fmt.Errorf("claude -p failed: %v (stderr: %s)", runErr, strings.TrimSpace(errOut))
	} else {
		raw := stdout.String()
		if len(raw) > 400 {
			raw = raw[:400]
		}
		return nil, fmt.Errorf("parse claude envelope (raw: %q)", raw)
	}

	// With --json-schema set, the model's structured answer lands in
	// `structured_output` and `result` is empty. Otherwise we fall back
	// to `result` (free-form text).
	content := env.Result
	if len(env.StructuredOutput) > 0 && string(env.StructuredOutput) != "null" {
		content = string(env.StructuredOutput)
	}

	return &openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{
			Index: 0,
			Message: openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: content,
			},
			FinishReason: openai.FinishReasonStop,
		}},
		Usage: openai.Usage{
			CompletionTokens: env.Usage.OutputTokens,
		},
	}, nil
}

func (b *ClaudeCLIBackend) Mode() string { return "claude-cli" }
