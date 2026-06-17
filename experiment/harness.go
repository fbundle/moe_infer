package main

import (
	"context"
	"errors"
	"fmt"
	"slices"

	openai "github.com/sashabaranov/go-openai"
)

// LoopExitReason — Go has no enums; typed string constants are the idiom.
// Set on History.Reason every time ChatCompletionLoop returns.
type LoopExitReason string

const (
	ExitOK              LoopExitReason = "ok"
	ExitInvalidInput    LoopExitReason = "invalid_input"
	ExitMaxRetries      LoopExitReason = "max_retries_reached"
	ExitBudgetExhausted LoopExitReason = "budget_exhausted"
	ExitCallError       LoopExitReason = "call_error" // completion func returned err/nil
)

// History records every message sent, every response received, and how
// the loop exited. Reason is always set after ChatCompletionLoop returns.
type History struct {
	Messages  []openai.ChatCompletionMessage
	Responses []openai.ChatCompletionResponse
	Reason    LoopExitReason
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
	validate func(string) (T, string, error), // (parsed, hint-for-LLM, err-for-caller)
	maxRetries int,
	maxCompletionTokens int,
) (o T, h History, err error) {
	if maxRetries < 0 {
		h.Reason = ExitInvalidInput
		return o, h, errors.New("maxTries must be non-negative")
	}
	if maxCompletionTokens <= 0 {
		h.Reason = ExitInvalidInput
		return o, h, errors.New("maxCompletionTokens must positive")
	}
	remainingTokens := maxCompletionTokens
	reaminingRetries := maxRetries
	h = History{
		Messages:  slices.Clone(initial),
		Responses: nil,
	}
	var lastValidateErr error // surfaced as err when we run out

	for {
		if remainingTokens == 0 {
			h.Reason = ExitBudgetExhausted
			return o, h, lastValidateErr
		}
		if reaminingRetries == 0 {
			h.Reason = ExitMaxRetries
			return o, h, lastValidateErr
		}
		// add budget hint
		h.Messages = append(h.Messages, budgetMessage(remainingTokens))
		// make call
		r, err := completion(ctx, h.Messages, remainingTokens)
		if err != nil || r == nil {
			// not recoverable
			h.Reason = ExitCallError
			return o, h, err
		}
		// make call success, update history
		h.Responses = append(h.Responses, *r)
		h.Messages = append(h.Messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: r.Choices[0].Message.Content,
		})
		// validate
		var hint string
		o, hint, err = validate(r.Choices[0].Message.Content)
		if err == nil { // validate success
			h.Reason = ExitOK
			return o, h, nil
		}
		// retry loop — hint is LLM-facing, err is caller-facing
		lastValidateErr = err
		h.Messages = append(h.Messages, errorMessage(hint))
		reaminingRetries -= 1
		remainingTokens -= r.Usage.CompletionTokens
	}
}
