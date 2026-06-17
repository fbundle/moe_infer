package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	openai "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

// History records every message sent and every response received.
type History struct {
	Messages  []openai.ChatCompletionMessage
	Responses []openai.ChatCompletionResponse
}

func budgetMessage(remainingTokens int) openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: fmt.Sprintf("[budget: %d tokens remaining for this response]", remainingTokens),
	}
}

func errorMessage(content string) openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: content,
	}
}

func ChatCompletionLoop[T any](
	ctx context.Context,
	initial []openai.ChatCompletionMessage,
	completion func(ctx context.Context, messages []openai.ChatCompletionMessage, maxCompletionTokens int) (*openai.ChatCompletionResponse, error),
	validate func(string) (T, error),
	maxRetries int,
	maxCompletionTokens int,
) (o T, h History, err error) {
	if maxRetries < 0 {
		return o, h, errors.New("maxTries must be non-negative")

	}
	if maxCompletionTokens <= 0 {
		return o, h, errors.New("maxCompletionTokens must positive")
	}
	remainingTokens := maxCompletionTokens
	reaminingRetries := maxRetries
	h = History{
		Messages:  slices.Clone(initial),
		Responses: nil,
	}

	for {
		if remainingTokens == 0 || reaminingRetries == 0 {
			return o, h, errors.New("loop break")
		}
		// add budget hint
		h.Messages = append(h.Messages, budgetMessage(remainingTokens))
		// make call
		r, err := completion(ctx, h.Messages, remainingTokens)
		if err != nil || r == nil {
			// not recoverable
			return o, h, err
		}
		// make call success, update history
		h.Responses = append(h.Responses, *r)
		h.Messages = append(h.Messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: r.Choices[0].Message.Content,
		})
		// validate
		o, err = validate(r.Choices[0].Message.Content)
		if err == nil { // validate success
			return o, h, nil
		}
		// retry loop
		h.Messages = append(h.Messages, errorMessage(err.Error()))
		reaminingRetries -= 1
		remainingTokens -= r.Usage.CompletionTokens
	}
}

// ── OpenAI-compatible backend (strict json_schema → json_object fallback) ──

type Backend struct {
	client   *openai.Client
	model    string
	mu       sync.Mutex
	strictOK *bool // nil = unknown
}

func NewBackend(baseURL, apiKey, model string) *Backend {
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = baseURL
	return &Backend{client: openai.NewClientWithConfig(cfg), model: model}
}

// CallWithSchema makes one chat-completion call enforcing a JSON schema.
// Tries strict json_schema first; falls back permanently to json_object
// when the endpoint refuses (e.g. DeepSeek, older mlx servers).
func (b *Backend) CallWithSchema(
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

func (b *Backend) setStrict(v bool) {
	b.mu.Lock()
	b.strictOK = &v
	b.mu.Unlock()
}
