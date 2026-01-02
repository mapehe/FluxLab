# ------------------------------------------------
# Generic Makefile for CUDA Projects (Enhanced for GTest + LibTorch)
# ------------------------------------------------

TORCH_PATH      = $(HOME)/libs/libtorch
# Use -isystem to ignore warnings inside Torch headers
TORCH_INCS      = -isystem $(TORCH_PATH)/include -isystem $(TORCH_PATH)/include/torch/csrc/api/include
# Linker flags: RPATH ensures the executable finds .so files at runtime
TORCH_RPATH     = -Xlinker -rpath=$(TORCH_PATH)/lib
TORCH_LIBS = -L$(TORCH_PATH)/lib \
             -ltorch \
             -Xlinker --no-as-needed -ltorch_cuda -Xlinker --as-needed \
             -lc10_cuda \
             -lc10 \
             -ltorch_cpu

# ------------------------------------------------
# Compiler settings
# ------------------------------------------------
NVCC          = nvcc
# Added _GLIBCXX_USE_CXX11_ABI=1 to force modern ABI
NVCC_FLAGS    = -O3 -std=c++20 --extended-lambda -diag-suppress 191 -Xcompiler -Wmissing-field-initializers -DCOMMIT_HASH=\"${COMMIT_HASH}\" -D_GLIBCXX_USE_CXX11_ABI=1
CXX_FLAGS     = -O3 -std=c++20 -Wmissing-field-initializers -DCOMMIT_HASH=\"${COMMIT_HASH}\" -D_GLIBCXX_USE_CXX11_ABI=1

# Append Torch Libs to standard LIBS
LIBS          = -lcufft

# Google Test Libraries
TEST_LIBS     = -lgtest -lgtest_main -lpthread -lgmock

# Dependency Generation Flags
DEP_FLAGS     = -MMD

# Project definitions
TARGET        = main
TEST_TARGET   = run_tests
SRC_DIR       = src
TEST_SRC_DIR  = src-test
OBJ_DIR       = obj
TEST_OBJ_DIR  = obj-test
BIN_DIR       = bin
INC_DIR       = include

# ------------------------------------------------
# Source & Object Discovery
# ------------------------------------------------

# 1. Application Sources
CU_SOURCES    = $(shell find $(SRC_DIR) -name "*.cu")
CPP_SOURCES   = $(shell find $(SRC_DIR) -name "*.cpp")

# 2. Test Sources
TEST_CU_SOURCES  = $(shell find $(TEST_SRC_DIR) -name "*.cu")
TEST_CPP_SOURCES = $(shell find $(TEST_SRC_DIR) -name "*.cpp")

# 3. Object Generation
CU_OBJECTS    = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))
CPP_OBJECTS   = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))
OBJECTS       = $(CU_OBJECTS) $(CPP_OBJECTS)

# Test Objects
TEST_CU_OBJECTS  = $(patsubst $(TEST_SRC_DIR)/%.cu, $(TEST_OBJ_DIR)/%.o, $(TEST_CU_SOURCES))
TEST_CPP_OBJECTS = $(patsubst $(TEST_SRC_DIR)/%.cpp, $(TEST_OBJ_DIR)/%.o, $(TEST_CPP_SOURCES))
TEST_OBJECTS     = $(TEST_CU_OBJECTS) $(TEST_CPP_OBJECTS)

# 4. Dependency Files
DEPS          = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.d, $(CU_SOURCES)) \
                $(patsubst $(TEST_SRC_DIR)/%.cu, $(TEST_OBJ_DIR)/%.d, $(TEST_CU_SOURCES))

# ------------------------------------------------
# Filter out main.o for testing
# ------------------------------------------------
APP_MAIN_OBJ  = $(OBJ_DIR)/main.o
LIB_OBJECTS   = $(filter-out $(APP_MAIN_OBJ), $(OBJECTS))

# ------------------------------------------------
# Build Rules
# ------------------------------------------------

all: directories $(BIN_DIR)/$(TARGET)

test: directories $(BIN_DIR)/$(TEST_TARGET)

# Link Application
# Added TORCH_RPATH here so the binary knows where libs are
$(BIN_DIR)/$(TARGET): $(OBJECTS)
	@echo "Linking Application..."
	$(NVCC) $(OBJECTS) \
		-L$(HOME)/libs/libtorch/lib \
		-ltorch \
		-Xlinker --no-as-needed -ltorch_cuda -lc10_cuda -Xlinker --as-needed \
		-lc10 -ltorch_cpu \
		$(LIBS) $(TORCH_RPATH) -o $@
	@echo "Build complete: $@"

# Link Tests
$(BIN_DIR)/$(TEST_TARGET): $(LIB_OBJECTS) $(TEST_OBJECTS)
	@echo "Linking Tests..."
	# Added $(TORCH_LIBS) here!
	$(NVCC) $(LIB_OBJECTS) $(TEST_OBJECTS) $(TORCH_LIBS) $(LIBS) $(TEST_LIBS) $(TORCH_RPATH) -o $@
	@echo "Test Build complete: $@"

# Compile App CUDA
# Added TORCH_INCS
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling CUDA $<..."
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(DEP_FLAGS) -I$(INC_DIR) $(TORCH_INCS) -c $< -o $@

# Compile App C++
# Added TORCH_INCS
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling C++ $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(DEP_FLAGS) -I$(INC_DIR) $(TORCH_INCS) -c $< -o $@

# Compile Test CUDA
# Added TORCH_INCS
$(TEST_OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cu
	@echo "Compiling Test CUDA $<..."
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(DEP_FLAGS) -I$(INC_DIR) $(TORCH_INCS) -c $< -o $@

# Compile Test C++
# Added TORCH_INCS
$(TEST_OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cpp
	@echo "Compiling Test C++ $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(DEP_FLAGS) -I$(INC_DIR) $(TORCH_INCS) -c $< -o $@

directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(TEST_OBJ_DIR)
	@mkdir -p $(BIN_DIR)

-include $(DEPS)

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(TEST_OBJ_DIR) $(BIN_DIR)

run: all
	@echo "Running App..."
	@./$(BIN_DIR)/$(TARGET)

run-test: test
	@echo "Running Tests..."
	@./$(BIN_DIR)/$(TEST_TARGET)

.PHONY: all clean run test run-test directories
