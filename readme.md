# DLBanchmarking playground 

Based on repo brain network transformer https://github.com/Wayfear/BrainNetworkTransformer
Brain Network Transformer is the open-source implementation of the NeurIPS 2022 paper [Brain Network Transformer](https://arxiv.org/abs/2210.06681).


## Dataset

Download the ABIDE dataset from [here](https://drive.google.com/file/d/14UGsikYH_SQ-d_GvY2Um2oEHw3WNxDY3/view?usp=sharing).

## Usage
Of course. Here is the `README.md` file formatted in Markdown.

---

# Brain Network Transformer Project

This project provides a flexible, configuration-driven framework for training and evaluating deep learning models on brain network data. It uses [Hydra](https://hydra.cc/) for configuration management and [MLflow](https://mlflow.org/) for experiment tracking.

## Quickstart

### 1. Setup the Environment

This project requires a specific set of libraries to ensure binary compatibility, especially on macOS.

First, create and activate a clean Python 3.11 environment using your preferred tool (e.g., `venv`, `conda`). Then, install all required packages from the `requirements.txt` file.

```bash
# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Launch the Experiment Tracker

This framework uses MLflow to log parameters, metrics, and model artifacts. To view your experiments, you need to run the MLflow UI.

In a separate terminal, navigate to the project root and run:
```bash
mlflow ui
```This will start a local server, typically at `http://127.0.0.1:5000`. Open this URL in your browser to see your experiment dashboard.

### 3. Run an Experiment

The entire training pipeline is controlled by YAML configuration files located in `source/conf/`. You can run experiments using the provided `run_mlp_model.py` script.


After each run, refresh the MLflow UI to see your new results, compare configurations, and inspect the saved model artifacts.

## Configuration System Overview

The behavior of the training pipeline is controlled by a set of modular YAML files.

-   `source/conf/config_mlp.yaml`: The main entry point that sets the default components for a run.
-   `source/conf/model/`: Contains configurations for different model architectures (e.g., `mlp.yaml`, `transformer.yaml`).
-   `source/conf/optimizer/`: Defines one or more optimizers to use (e.g., `adamw_full.yaml`).
-   `source/conf/preprocess/`: Defines the data preprocessing and augmentation pipelines (e.g., `mlp_transform.yaml`).
-   `source/conf/training/`: Contains training loop settings like epochs and early stopping patience (e.g., `basic_supervised.yaml`).

By mixing and matching these configuration files from the command line, you can easily define and execute a wide variety of experiments.

## Example Script (`run_mlp_model.py`)

This script, located in the project root, serves as the main entry point for running experiments.

