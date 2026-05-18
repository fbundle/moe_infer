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
ACCELERATE ?= 1
ifeq ($(ACCELERATE),1)
	CFLAGS += -DACCELERATE_NEW_LAPACK
endif

# ── Debug build ───────────────────────────────────────────────────────────
DEBUG ?= 0
ifeq ($(DEBUG),1)
	CFLAGS := -g -O0 -Wall -Wextra -fobjc-arc -Isrc -DDEBUG
endif

# ── Targets ───────────────────────────────────────────────────────────────
.PHONY: all clean metallib help

all: $(BIN_DIR)/infer

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# ── Inference engine ──────────────────────────────────────────────────────
$(BIN_DIR)/infer: $(SRC_DIR)/infer.m $(SRC_DIR)/shaders.metal $(SRC_DIR)/model_config.h $(SRC_DIR)/config.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(FW) $(LDFLAGS) $(SRC_DIR)/infer.m -o $@

# ── Pre-compiled Metal library ────────────────────────────────────────────
metallib: $(BIN_DIR)/shaders.metallib

$(BIN_DIR)/shaders.metallib: $(SRC_DIR)/shaders.metal | $(BIN_DIR)
	$(METALC) -c $(SRC_DIR)/shaders.metal -o $(BIN_DIR)/shaders.air
	$(METALLIB) $(BIN_DIR)/shaders.air -o $@

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BIN_DIR)

# ── Help ──────────────────────────────────────────────────────────────────
help:
	@echo "Flash-MoE Makefile"
	@echo ""
	@echo "Build targets:"
	@echo "  make              Build infer"
	@echo "  make metallib     Pre-compile Metal shader library"
	@echo "  make clean        Remove bin/"
	@echo ""
	@echo "Build options:"
	@echo "  make DEBUG=1      Debug build"
	@echo "  make ACCELERATE=0 Build without Accelerate BLAS"
