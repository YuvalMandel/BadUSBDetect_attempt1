# ==============================================================================
# VARIABLES (Paths and Commands)
# ==============================================================================
PYTHON = python -X utf8
DATA_DIR = dataset_generator
MLP_DIR = MLP
REGRESSOR_DIR = $(MLP_DIR)/regressor
GRU_DIR = GRU
HTM_DIR = HTM
RESULTS_DIR = results

# Folders for generated data
BOTS_TRAIN = $(DATA_DIR)/Synthetic_Bots
BOTS_TEST = $(DATA_DIR)/Synthetic_Bots_test
HUMANS_TRAIN = $(DATA_DIR)/Balanced_Humans
HUMANS_TEST = $(DATA_DIR)/Balanced_Humans_test

SPLIT_JSON = data_split.json

.PHONY: help generate_dataset split_persons train_mlp train_gru train_htm test_htm clean_results clean_dataset clean_models clean_all

# ==============================================================================
# HELP: USER INSTRUCTIONS
# ==============================================================================
help:
	@echo "======================================================================"
	@echo "                BadUSB Detector - Makefile Pipeline                   "
	@echo "======================================================================"
	@echo "Step-by-step instructions for the full training cycle:"
	@echo ""
	@echo "1. Generate bot/human files:"
	@echo "   make generate_dataset"
	@echo ""
	@echo "2. Create person-disjoint split manifest (data_split.json):"
	@echo "   make split_persons"
	@echo ""
	@echo "3. Train MLP (Train regressor -> Extract features -> Train MLP):"
	@echo "   make train_mlp"
	@echo ""
	@echo "4. Train GRU (Assemble tensors -> Train GRU):"
	@echo "   make train_gru"
	@echo ""
	@echo "5. Train HTM (Prepare data -> Train HTM):"
	@echo "   make train_htm"
	@echo ""
	@echo "6. Test HTM model:"
	@echo "   make test_htm"
	@echo ""
	@echo "----------------------------------------------------------------------"
	@echo "Cleanup Utilities:"
	@echo "  make clean_results  - Remove charts (PNG) and reports (CSV) from results"
	@echo "  make clean_dataset  - Remove generated bot/human data and CSV/PT/PKL files"
	@echo "  make clean_models   - Remove model weights (.pth, .pkl, .npy, .npz)"
	@echo "  make clean_all      - Perform full project cleanup"
	@echo "======================================================================"

# ==============================================================================
# DATA PREPARATION
# ==============================================================================
generate_dataset:
	@echo "--- 1. Generating training bots ---"
	cd $(DATA_DIR) && $(PYTHON) bot_generator.py -o Synthetic_Bots -f 25 -e 80
	@echo "--- 2. Generating test bots (for regressor) ---"
	cd $(DATA_DIR) && $(PYTHON) bot_generator.py -o Synthetic_Bots_test -f 5 -e 80
	@echo "--- 3. Collecting training humans (ending in 1.txt) ---"
	cd $(DATA_DIR) && $(PYTHON) human_generator.py -o Balanced_Humans -f 124 -l 80 -e 1
	@echo "--- 4. Collecting test humans (for regressor only) ---"
	cd $(DATA_DIR) && $(PYTHON) human_generator.py -o Balanced_Humans_test -f 24 -l 80 -e 0
	@echo "✅ Data generation complete!"

# Create person-disjoint train/val/test manifest from s2/ raw files
split_persons:
	@echo "--- Creating person-disjoint split (data_split.json) ---"
	$(PYTHON) split_persons.py --bots-dir $(BOTS_TRAIN)
	@echo "✅ Split manifest saved to $(SPLIT_JSON)"

# ==============================================================================
# MLP AND REGRESSOR TRAINING
# ==============================================================================
train_mlp:
	@echo "--- 1. Training Polynomial Regressor ---"
	cd $(REGRESSOR_DIR) && $(PYTHON) regressor_train.py -hu ../../$(HUMANS_TRAIN) -m poly_regressor.pkl
	@echo "--- 2. Testing Regressor ---"
	cd $(REGRESSOR_DIR) && $(PYTHON) test_regressor.py -hu ../../$(HUMANS_TEST) -b ../../$(BOTS_TEST) -m poly_regressor.pkl
	@echo "--- 3. Copying regressor model to MLP root ---"
	cp $(REGRESSOR_DIR)/poly_regressor.pkl $(MLP_DIR)/poly_regressor.pkl
	@echo "--- 4. Generating person-disjoint CSV datasets (Feature Extraction) ---"
	cd $(MLP_DIR) && $(PYTHON) dataset_csv_generator.py --split-json ../$(SPLIT_JSON)
	@echo "--- 5. Training MLP model ---"
	cd $(MLP_DIR) && $(PYTHON) model_training.py
	@echo "✅ MLP training successfully completed!"

# ==============================================================================
# GRU TRAINING
# ==============================================================================
train_gru:
	@echo "--- 1. Converting data to Tensors (PT) ---"
	cd $(GRU_DIR) && $(PYTHON) translate_to_tensors.py --split-json ../$(SPLIT_JSON)
	@echo "--- 2. Training GRU model ---"
	cd $(GRU_DIR) && $(PYTHON) train_gru.py
	@echo "✅ GRU training successfully completed!"

# ==============================================================================
# HTM TRAINING
# ==============================================================================
train_htm:
	@echo "--- 1. Preparing HTM data (windows_cache.pkl) ---"
	cd $(HTM_DIR) && $(PYTHON) htm_prepare_data.py
	@echo "--- 2. Training HTM model ---"
	cd $(HTM_DIR) && $(PYTHON) htm_train.py
	@echo "✅ HTM training successfully completed!"

test_htm:
	@echo "--- Testing HTM model ---"
	-@LATEST_MODEL=$(shell ls -t $(RESULTS_DIR)/HTM/models/*.pkl 2>/dev/null | head -n 1)
	@if [ -z "$(LATEST_MODEL)" ]; then \
		echo "ERROR: No HTM models found in $(RESULTS_DIR)/HTM/models/"; \
		exit 1; \
	fi; \
	echo "Using latest model: $${LATEST_MODEL}"; \
	$(PYTHON) $(HTM_DIR)/htm_test.py --model "$${LATEST_MODEL}"

# ==============================================================================
# CLEANUP
# ==============================================================================
clean_results:
	@echo "🧹 Cleaning up charts and reports..."
	rm -rf $(RESULTS_DIR)/*

clean_dataset:
	@echo "Cleaning up generated datasets..."
	rm -rf $(BOTS_TRAIN)
	rm -rf $(BOTS_TEST)
	rm -rf $(HUMANS_TRAIN)
	rm -rf $(HUMANS_TEST)
	rm -f $(SPLIT_JSON)
	rm -f $(MLP_DIR)/train_dataset.csv
	rm -f $(MLP_DIR)/val_dataset.csv
	rm -f $(MLP_DIR)/test_dataset.csv
	rm -f $(GRU_DIR)/rnn_dataset.pt
	rm -f $(HTM_DIR)/windows_cache.pkl

clean_models:
	@echo "🧹 Cleaning up saved weights and parameters..."
	rm -f $(MLP_DIR)/badusb_model.pth
	rm -f $(MLP_DIR)/scaler_params.npy
	rm -f $(MLP_DIR)/reference_pool.npz
	rm -f $(MLP_DIR)/poly_regressor.pkl
	rm -f $(REGRESSOR_DIR)/poly_regressor.pkl
	rm -f $(GRU_DIR)/gru_model.pth
	rm -f $(GRU_DIR)/rnn_scaler_params.npy
	rm -rf $(RESULTS_DIR)/HTM/models/*

clean_all: clean_results clean_dataset clean_models
	@echo "✨ Full project cleanup complete!"
