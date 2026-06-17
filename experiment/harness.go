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
	ExitMaxSteps        LoopExitReason = "max_steps_reached"
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

func envMessage(content string) openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: content,
	}
}

// ChatCompletionLoop drives an agent loop. The caller supplies a `step`
// function that the loop calls with the assistant's latest content. Step
// either returns a final value (isFinal=true) or an intermediate
// environment message (isFinal=false) which gets fed back as a system
// message for the next turn.
//
// Same signature handles single-shot benches (step parses output; isFinal
// on valid parse, retry-hint on parse failure) and rollouts (step
// advances env state, returns intermediate "round t observation" or
// final summary).
//
// maxSteps is a safety cap — loop aborts with ExitMaxSteps if step never
// returns isFinal within that many calls.
func ChatCompletionLoop[T any](
	ctx context.Context,
	initial []openai.ChatCompletionMessage,
	completion func(ctx context.Context, messages []openai.ChatCompletionMessage, maxCompletionTokens int) (*openai.ChatCompletionResponse, error),
	toAssistantMessage func(openai.ChatCompletionResponse) openai.ChatCompletionMessage,
	step func(content string) (final T, hint string, isFinal bool),
	maxSteps int,
	maxCompletionTokens int,
) (o T, h History, err error) {
	if maxSteps <= 0 {
		h.Reason = ExitInvalidInput
		return o, h, errors.New("maxSteps must be positive")
	}
	if maxCompletionTokens <= 0 {
		h.Reason = ExitInvalidInput
		return o, h, errors.New("maxCompletionTokens must be positive")
	}
	remainingTokens := maxCompletionTokens
	remainingSteps := maxSteps
	h = History{Messages: slices.Clone(initial)}

	for {
		if remainingTokens == 0 {
			h.Reason = ExitBudgetExhausted
			return o, h, nil
		}
		if remainingSteps == 0 {
			h.Reason = ExitMaxSteps
			return o, h, nil
		}
		h.Messages = append(h.Messages, budgetMessage(remainingTokens))
		r, err := completion(ctx, h.Messages, remainingTokens)
		if err != nil || r == nil {
			h.Reason = ExitCallError
			return o, h, err
		}
		h.Responses = append(h.Responses, *r)
		h.Messages = append(h.Messages, toAssistantMessage(*r))

		final, hint, isFinal := step(r.Choices[0].Message.Content)
		if isFinal {
			h.Reason = ExitOK
			return final, h, nil
		}
		// Intermediate: feed hint back as a system message and continue.
		h.Messages = append(h.Messages, envMessage(hint))
		remainingSteps--
		remainingTokens -= r.Usage.CompletionTokens
	}
}
