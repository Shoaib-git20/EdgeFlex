## Vision Transformer Pruning and Quantization Pipeline

This repository contains a complete pipeline to compress Vision Transformer (ViT) models for deployment on memory-constrained edge devices. It includes:

- **Activation Profiling**
- **Memory-Aware Pruning (MLP focused)**
- **Mixed Precision Execution (FP16)**
- **Activation-Aware Quantization (AWQ)**

## Notebook

- 📄 `vit_prune_quant_pipeline.ipynb`: Main pipeline notebook for profiling, pruning, quantization, and evaluation.

## Model

- Model: **ViT-Huge**  
  - Layers: 32  
  - Hidden Size: 1280  
  - MLP Size: 5120  
  - Heads: 16  
  - Parameters: 632M 

## Getting Started

These instructions will help you set up and run the notebook end-to-end on your local machine.

## 1. Clone the Repository
git clone https://github.com/your-username/vit-prune-quant.git
cd vit-prune-quant

## 2. Set Up Python Environment
We recommend using a virtual environment:
python3 -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

## 3. Install Requirements
Once the environment is activated, install dependencies:
pip install -r requirements.txt

## 4. Run the Notebook
Launch Jupyter and open the notebook:
jupyter notebook

Then open vit_prune_quant_pipeline.ipynb and execute the cells in order.

## 5. Dataset
The CIFAR-10 dataset will be downloaded automatically by the notebook using torchvision.datasets.CIFAR10.

## Notes
- The notebook assumes CUDA is available. Make sure you are running on a GPU-enabled machine for best performance.
- You can modify batch size, dataset path, or calibration size in the early cells of the notebook.
