EXE ?= fatshashcorchess0
BUILD_DIR ?= build-openbench
TARGET ?= fatshashcorchess0
CMAKE ?= cmake

.PHONY: all clean

all:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(CXX)
	$(CMAKE) --build $(BUILD_DIR) --config Release
	@if [ -f "$(BUILD_DIR)/$(TARGET)" ]; then cp "$(BUILD_DIR)/$(TARGET)" "$(EXE)"; \
	elif [ -f "$(BUILD_DIR)/Release/$(TARGET)" ]; then cp "$(BUILD_DIR)/Release/$(TARGET)" "$(EXE)"; \
	elif [ -f "$(BUILD_DIR)/$(TARGET).exe" ]; then cp "$(BUILD_DIR)/$(TARGET).exe" "$(EXE)"; \
	elif [ -f "$(BUILD_DIR)/Release/$(TARGET).exe" ]; then cp "$(BUILD_DIR)/Release/$(TARGET).exe" "$(EXE)"; \
	else echo "Could not locate built binary for $(TARGET)"; exit 1; fi

clean:
	-$(CMAKE) --build $(BUILD_DIR) --config Release --target clean
