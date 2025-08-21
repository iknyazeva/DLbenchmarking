#!/bin/bash

# ===================================================================================
# A Bash script to run ablation studies for the BrainNetworkTransformer model.
#
# This script launches a series of experiments by calling 'run_bnttransformer.py'
# with different command-line parameter overrides. Each study is logged to MLflow.
#
# Usage:
#    chmod +x run_bnt_ablations.sh
#   ./run_bnt_ablations.sh
# ===================================================================================

# --- Configuration ---

# The main Python script to execute.
PYTHON_SCRIPT="run_brainnetcnn_model.py"

# --- Helper function for logging ---
log() {
  echo ""
  echo "################################################################################"
  echo "# $1"
  echo "################################################################################"
  echo ""
}


# --- Script Execution ---

# Check if the main script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Main script '$PYTHON_SCRIPT' not found."
    echo "Please make sure you are in the project's root directory."
    exit 1
fi



# --- Ablation Study 1 ---

log "Starting Ablation Study 1: Effect Shuffled FC Matrix"

echo "Running with NO shuffling..."
python $PYTHON_SCRIPT dataset.shuffle_matrix=False

echo "Running with shuffling..."
python $PYTHON_SCRIPT dataset.shuffle_matrix=True

echo "Ablation Study 1 finished."


# --- Ablation Study 2 ---
log "Starting Ablation Study 2: Model Architecture"

echo "Running original model"
python $PYTHON_SCRIPT

echo "Running smaller model"
python $PYTHON_SCRIPT model.num_E2Eblock=1 model.E2E_channels="[16]" model.E2N_channels=32 model.N2G_channels=128 model.hidden_out_Linear="[128, 2]"

echo "Running original sized model with mistake"
python $PYTHON_SCRIPT model.num_E2Eblock=2 model.E2E_channels="[32, 32]" model.E2N_channels=1 model.N2G_channels=256 model.hidden_out_Linear="[256, 128, 2]"

echo "Ablation Study 2 finished."


log "All ablation studies have been launched!"
echo "Check your MLflow UI to compare the results of all runs."