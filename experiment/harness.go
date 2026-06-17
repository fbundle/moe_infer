package main

import (
	"context"
	"errors"
	"fmt"

	"slices"

	openai "github.com/sashabaranov/go-openai"
)

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
