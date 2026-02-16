# Comparative Study of Deep Learning Algorithms for Sentiment Analysis

This repository contains the research and implementation of a multi-label sentiment analysis system using **BERT** and **Ensemble learning** techniques.

## Overview
The project evaluates the performance of Transformer-based models in detecting complex human emotions. By utilizing the **GoEmotions** dataset and fine-tuning a **BERT-base-uncased** architecture, the system classifies text into 28 distinct emotional categories.

## Repository Structure
* **`Train_bert_original.ipynb`**: The core training pipeline, including:
    * Data preprocessing & custom PyTorch `DataModule`.
    * Model architecture based on `BertForSequenceClassification`.
    * Training logic with Early Stopping and Learning Rate scheduling.
* **`Eval_bert_model.ipynb`**: The evaluation suite used to:
    * Load pre-trained model checkpoints.
    * Generate detailed classification reports.
    * Calculate performance metrics (AUROC, F1-score) for all labels.

## Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** PyTorch, PyTorch Lightning
* **NLP Library:** Hugging Face Transformers
* **Metrics:** Torchmetrics, Scikit-learn
* **Environment:** Google Colab / Jupyter Notebooks

## Key Features
* **Multi-label Classification:** Simultaneously detects overlapping emotions in a single text snippet.
* **Transfer Learning:** Fine-tuned BERT for high-dimensional text embeddings.
* **Ensemble Strategy:** Combines model predictions to improve robustness, especially for minority emotion classes.
* **Performance:** Achieved a competitive average **AUROC of ~0.93** across major emotion categories.

## Setup & Usage
1. **Clone the repository:**
   ```bash
   git clone (https://github.com/sthermiotis00/bert-thesis)
2. **Install required packages:**
    ```bash
   pip install torch transformers pytorch-lightning torchmetrics pandas numpy
4. **Run the Notebooks:**
    ```bash
   Execute Train_bert_original.ipynb to train the model from scratch.
   Execute Eval_bert_model.ipynb to run inference and view results.
