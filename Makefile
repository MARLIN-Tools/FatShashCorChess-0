EXE ?= fatshashcorchess0
BUILD_DIR ?= build-openbench
TARGET ?= fatshashcorchess0
CMAKE ?= cmake

.PHONY: all clean

all:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(CXX)
	$(CMAKE) --build $(BUILD_DIR) --config Release
	$(CMAKE) -E copy_if_different "$(BUILD_DIR)/Release/$(TARGET).exe" "$(EXE).exe"

clean:
	-$(CMAKE) --build $(BUILD_DIR) --config Release --target clean
