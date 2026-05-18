# Flash-MoE — Pure C/Metal MoE inference engine
# Builds from src/ into bin/

CC       = clang
CFLAGS   = -O2 -Wall -Wextra -fobjc-arc -Isrc
LDFLAGS  = -lpthread -lcompression
FW       = -framework Metal -framework Foundation -framework Accelerate
METALC   = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

SRC_DIR  = src
BIN_DIR  = bin

# ── Accelerate BLAS for GatedDeltaNet recurrence ──────────────────────────
# Paper finding: BLAS delta-net → 64% faster attention (0.78ms → 0.28ms).
# Disable with ACCELERATE=0 if linking against older macOS SDKs.
ACCELERATE ?= 1
ifeq ($(ACCELERATE),1)
	CFLAGS += -DACCELERATE_NEW_LAPACK
endif

# ── Debug build ───────────────────────────────────────────────────────────
# make DEBUG=1 — adds -g, disables optimizations, enables debug checks.
DEBUG ?= 0
ifeq ($(DEBUG),1)
	CFLAGS := -g -O0 -Wall -Wextra -fobjc-arc -Isrc -DDEBUG
endif

# ── Targets ───────────────────────────────────────────────────────────────
.PHONY: all clean metallib infer chat benchmark test

all: $(BIN_DIR)/infer $(BIN_DIR)/chat

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# ── Inference engine (infer.m) ────────────────────────────────────────────
$(BIN_DIR)/infer: $(SRC_DIR)/infer.m $(SRC_DIR)/shaders.metal $(SRC_DIR)/model_config.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(FW) $(LDFLAGS) $(SRC_DIR)/infer.m -o $@

# ── MoE benchmark (main.m) ────────────────────────────────────────────────
$(BIN_DIR)/bench: $(SRC_DIR)/main.m $(SRC_DIR)/shaders.metal $(SRC_DIR)/model_config.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(FW) $(LDFLAGS) $(SRC_DIR)/main.m -o $@

# ── Interactive chat TUI (chat.m) ─────────────────────────────────────────
$(BIN_DIR)/chat: $(SRC_DIR)/chat.m $(SRC_DIR)/linenoise.c $(SRC_DIR)/linenoise.h | $(BIN_DIR)
	$(CC) -O2 -Wall -fobjc-arc -framework Foundation $(SRC_DIR)/chat.m $(SRC_DIR)/linenoise.c -o $@

# ── Pre-compiled Metal library (optional — runtime compilation is default) ─
metallib: $(BIN_DIR)/shaders.metallib

$(BIN_DIR)/shaders.metallib: $(SRC_DIR)/shaders.metal | $(BIN_DIR)
	$(METALC) -c $(SRC_DIR)/shaders.metal -o $(BIN_DIR)/shaders.air
	$(METALLIB) $(BIN_DIR)/shaders.air -o $@

# ── Inference targets ─────────────────────────────────────────────────────
# Default: K=8 experts, 4-bit quantized (best quality)
infer: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 20

# K=4 experts (faster, may degrade quality on 8-expert models)
infer-k4: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 20 --k 4

# K=2 experts (fastest, lower quality)
infer-k2: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 20 --k 2

# ── 2-bit inference (faster, breaks JSON/tool calling — paper §2-bit) ─────
# Requires: packed_experts_2bit/ directory (run repack_experts_2bit.py)
infer-2bit: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 20 --2bit

# ── Cache mode experiments (paper §Trust the OS) ───────────────────────────
# Trust OS page cache (default, best): no flags needed
# This was the winning strategy — 38% faster than custom Metal LRU cache.

# Malloc expert cache (paper §Discarded: slower than OS page cache)
# Example: 2581 entries ≈ 17GB for ~80% hit rate on 397B model.
infer-malloc-cache: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 20 --malloc-cache 2581

# Expert LRU cache (paper §Discarded: Metal LRU)
infer-lru-cache: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 20 --cache-entries 2500

# ── CPU linear attention (paper §BLAS delta-net) ──────────────────────────
# Disables fused GPU delta-net; uses CPU Accelerate BLAS instead.
# 64% faster than scalar code, but GPU path is better overall.
infer-cpu-linear: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 20 --cpu-linear

# ── Expert prediction (paper §Discarded: -18%) ────────────────────────────
infer-predict: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 20 --predict

# ── Per-layer timing breakdown ────────────────────────────────────────────
infer-timing: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 20 --timing

# ── Cache telemetry (hit rate, reuse distance) ────────────────────────────
infer-telemetry: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 100 --cache-telemetry

# ── Expert frequency tracking ─────────────────────────────────────────────
infer-freq: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --prompt "Hello" --tokens 100 --freq

# ── LZ4 compressed experts (paper §Discarded: -13%) ────────────────────────
# Requires: packed_experts_lz4/ directory (run repack_experts_lz4.c)
# Auto-detected at runtime if directory exists.

# ── HTTP server mode (OpenAI-compatible API) ──────────────────────────────
serve: $(BIN_DIR)/infer
	./$(BIN_DIR)/infer --serve 8080

# ── Benchmark targets ─────────────────────────────────────────────────────
bench: $(BIN_DIR)/bench
	./$(BIN_DIR)/bench --fast --full --k 4 --benchmark

# MoE single-layer bench
bench-moe: $(BIN_DIR)/bench
	./$(BIN_DIR)/bench --layer 0 --fast --moe --benchmark

# Single expert bench
bench-expert: $(BIN_DIR)/bench
	./$(BIN_DIR)/bench --layer 0 --expert 0 --fast --benchmark

# ── Verification (Metal vs CPU reference) ─────────────────────────────────
verify: $(BIN_DIR)/bench
	./$(BIN_DIR)/bench --layer 0 --expert 0 --fast --verify

# ── Chat TUI ──────────────────────────────────────────────────────────────
chat: $(BIN_DIR)/chat
	./$(BIN_DIR)/chat --k 8

chat-k4: $(BIN_DIR)/chat
	./$(BIN_DIR)/chat --k 4

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BIN_DIR)

# ── Help ──────────────────────────────────────────────────────────────────
help:
	@echo "Flash-MoE Makefile"
	@echo ""
	@echo "Build:"
	@echo "  make              Build infer + chat"
	@echo "  make metallib     Pre-compile Metal shader library"
	@echo "  make clean        Remove bin/"
	@echo ""
	@echo "Inference (varying K):"
	@echo "  make infer        K=8 (best quality, default)"
	@echo "  make infer-k4     K=4 (faster, slight quality loss)"
	@echo "  make infer-k2     K=2 (fastest, lower quality)"
	@echo ""
	@echo "Quantization:"
	@echo "  make infer-2bit   2-bit experts (faster, breaks tool calling)"
	@echo ""
	@echo "Cache strategies (paper experiments):"
	@echo "  make infer              Trust OS page cache (default, best)"
	@echo "  make infer-malloc-cache  Malloc cache (paper: slower)"
	@echo "  make infer-lru-cache     Metal LRU cache (paper: -38%)"
	@echo ""
	@echo "Attention modes:"
	@echo "  make infer-cpu-linear  CPU BLAS delta-net (paper: 64% attn speedup)"
	@echo ""
	@echo "Experimental (paper: discarded):"
	@echo "  make infer-predict     Temporal expert prediction (-18%)"
	@echo ""
	@echo "Analysis:"
	@echo "  make infer-timing      Per-layer timing breakdown"
	@echo "  make infer-telemetry   Cache hit rate & reuse distance"
	@echo "  make infer-freq        Expert frequency tracking"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench             Full MoE benchmark"
	@echo "  make bench-moe         Single-layer MoE"
	@echo "  make bench-expert      Single expert"
	@echo "  make verify            Metal vs CPU correctness check"
	@echo ""
	@echo "Server:"
	@echo "  make serve             HTTP API on :8080"
	@echo ""
	@echo "Build options:"
	@echo "  make DEBUG=1           Debug build"
	@echo "  make ACCELERATE=0      Build without Accelerate BLAS"
