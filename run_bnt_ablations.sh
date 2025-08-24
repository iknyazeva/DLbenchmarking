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
PYTHON_SCRIPT="run_bnttransformer.py"

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



# --- Ablation Study 1: Effect of Different Positional Encodings ---
# We run the model three times, swapping out the entire pos_encoding block.
log "Starting Ablation Study 1: Effect of atlas"

echo "Running with ABIDE..."
#python $PYTHON_SCRIPT model.pos_encoding.name=none
python $PYTHON_SCRIPT dataset=ABIDE

echo "Running with ABIDE SCH..."
#python $PYTHON_SCRIPT model.pos_encoding.name=none
python $PYTHON_SCRIPT dataset=ABIDE_SCH model.pos_encoding.embed_dim=34 

log "Starting Ablation Study 1: Effect of atlas: round 2"


echo "Running with ABIDE..."
#python $PYTHON_SCRIPT model.pos_encoding.name=none
python $PYTHON_SCRIPT dataset=ABIDE

echo "Running with ABIDE SCH..."
#python $PYTHON_SCRIPT model.pos_encoding.name=none
python $PYTHON_SCRIPT dataset=ABIDE_SCH model.pos_encoding.embed_dim=34 

#echo "Running with IDENTITY (learnable) positional encoding..."
#python $PYTHON_SCRIPT model.pos_encoding.name=IdentityEncoding 

#echo "Running with RRWP (random walk) positional encoding..."
#python $PYTHON_SCRIPT model.pos_encoding.name=RRWPEncoding 

echo "Ablation Study 1 finished."


# --- Ablation Study 2: Effect of Transformer Hidden Dimension ---
# This study tests the size of the feed-forward network inside the transformer.
#log "Starting Ablation Study 2: Transformer Feed-Forward Dimension"

#echo "Running with dim_feedforward=256..."
#python $PYTHON_SCRIPT model.dim_feedforward=256

#echo "Running with dim_feedforward=512..."
#python $PYTHON_SCRIPT model.dim_feedforward=512

#echo "Running with dim_feedforward=1024..."
#python $PYTHON_SCRIPT model.dim_feedforward=1024

#echo "Ablation Study 2 finished."


# --- Ablation Study 3: Effect of Preprocessing  ---
# This study tests the size of the feed-forward network inside the transformer.
log "Starting Ablation Study 2: Augmentation"

echo "Running with no augmentation..."
python $PYTHON_SCRIPT preprocess=default

echo "Running with inclass_node_mixup"
python $PYTHON_SCRIPT preprocess=default dataset=ABIDE_SCH model.pos_encoding.embed_dim=34 


echo "Ablation Study 2 finished."

log "Starting Ablation Study 2: Augmentation: round 2"

echo "Running with no augmentation..."
python $PYTHON_SCRIPT preprocess=default

echo "Running with inclass_node_mixup"
python $PYTHON_SCRIPT preprocess=default dataset=ABIDE_SCH model.pos_encoding.embed_dim=34 


echo "Ablation Study 2 finished."



log "All ablation studies have been launched!"
echo "Check your MLflow UI to compare the results of all runs."