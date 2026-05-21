This repository contains the code and research for an end-to-end Machine Learning pipeline that performs binary sentiment analysis on the **IMDB Large Movie Review Dataset**. 

The primary objective of this project was to successfully fine-tune a modern Transformer model (`distilbert-base-uncased`) under strict hardware constraints (a standard CPU environment with 8GB RAM) and compare its performance against a classical Long Short-Term Memory (LSTM) baseline model. 

### Key Engineering Highlights:
* **Hardware Optimization:** Implemented **Layer Freezing** (freezing the bottom 4 transformer layers) and **Gradient Accumulation** (simulating a batch size of 16 using a physical batch size of 4) to prevent Out-of-Memory (OOM) errors on CPU.
* **Classical Baseline:** Built a PyTorch LSTM model using 100-dimensional GloVe embeddings from scratch for comparative analysis.
* **Model Interpretability:** Extracted and visualized the `[CLS]` token's self-attention weights to map exactly which sentiment-bearing words the model focused on to make its predictions.
* **Ablation Study:** Conducted a data-scaling ablation study (10%, 25%, 50%, 100%) to chart the non-linear trade-offs between predictive accuracy and computational training time.

---

## 📁 Repository Structure

The project was developed iteratively over four phases, documented in the following sequential Jupyter Notebooks:

* `01_preprocessing.ipynb`: Tokenization, padding/truncation (max 512 tokens), and caching datasets to disk.
* `02_finetuning.ipynb`: Freezing DistilBERT layers, configuring the Hugging Face `Trainer` API, and executing CPU-optimized training.
* `03_lstm_baseline_and_attention.ipynb`: Construction of the GloVe-LSTM baseline and extraction/visualization of transformer attention weights via heatmaps.
* `04_evaluation_and_ablation.ipynb`: Error analysis (extracting False Positives/Negatives), generating confusion matrices, and the data-scaling ablation study.
