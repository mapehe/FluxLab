# ------------------------------------------------
# Generic Makefile for CUDA Projects
# ------------------------------------------------

# Compiler settings
NVCC        = nvcc
NVCC_FLAGS  = -O3 -std=c++14
# Add architecture flags here (e.g., -arch=sm_75) if you know your GPU
# NVCC_FLAGS += -arch=sm_70 

# Project definitions
TARGET      = hello
SRC_DIR     = src
OBJ_DIR     = obj
BIN_DIR     = bin
INC_DIR     = include

# Find all .cu files in the src directory
SOURCES     = $(wildcard $(SRC_DIR)/*.cu)
# Create a list of object files based on source files
OBJECTS     = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SOURCES))

# ------------------------------------------------
# Build Rules
# ------------------------------------------------

# Default target: build the executable
all: directories $(BIN_DIR)/$(TARGET)

# Rule to link object files into the final executable
$(BIN_DIR)/$(TARGET): $(OBJECTS)
	@echo "Linking..."
	$(NVCC) $(OBJECTS) -o $@
	@echo "Build complete: $@"

# Rule to compile .cu files into .o files
# $< refers to the source file, $@ refers to the object file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) -c $< -o $@

# Create specific directories if they don't exist
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Clean up build artifacts
clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run the program
run: all
	@echo "Running $(TARGET)..."
	@./$(BIN_DIR)/$(TARGET)

# Phony targets help avoid conflicts with files named 'clean' or 'all'
.PHONY: all clean run directories
