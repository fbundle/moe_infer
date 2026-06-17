package main

import (
	"context"
	"strings"
	"sync"

	openai "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

// OpenAI-compatible backend (strict json_schema → json_object fallback).

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

// Mode reports what the backend is doing now: "strict", "json_object",
// or "unknown" before the first call decides.
func (b *Backend) Mode() string {
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
