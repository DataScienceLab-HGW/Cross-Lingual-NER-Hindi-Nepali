# Cross-Lingual Named Entity Recognition between Hindi and Nepali

This repository hosts the code and resources for conducting cross-lingual transfer learning experiments focused on Named Entity Recognition (NER) between Hindi and Nepali. This work forms part of a research study aimed at evaluating the effectiveness of pre-trained multilingual BERT models in monolingual and cross-lingual contexts, focusing on the linguistic similarities and differences between Hindi and Nepali.

## Overview

The core objective of this project is to explore the potential of cross-lingual transfer learning to improve NER performance in a resource-limited language like Nepali by leveraging models pre-trained on a relatively resource-abundant language such as Hindi. The experiments were conducted using the following pre-trained models:

- **[BERT Multilingual](https://huggingface.co/google-bert/bert-base-multilingual-cased)**
- **[DistilBERT Multilingual](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)**
- **[RemBERT](https://huggingface.co/google/rembert)**
- **[MuRIL](https://huggingface.co/google/muril-base-cased)**

These models were fine-tuned and evaluated on NER tasks using datasets specific to both Hindi and Nepali. The experiments involved monolingual evaluations to gauge performance within each language and cross-lingual evaluations to assess the models' ability to transfer linguistic knowledge across languages.

## Repository Structure

- `crosslingual_ner_hindi_nepali.ipynb`: The main notebook for conducting the experiments and analyzing the results.
- `trainer.py`: Script dedicated to training the models on the NER datasets, including functions for fine-tuning and evaluation.
- `requirements.txt`: A list of the Python packages required to run the experiments.

## Installation

To set up the environment, install Python 3.8 or later. The required packages can be installed by executing:

```bash
pip install -r requirements.txt
```
The primary dependencies include:
- **PyTorch** (with CUDA support) - for model training and inference.
- **Hugging Face Transformers** - to leverage pre-trained BERT models and fine-tune them for NER tasks.
- **TensorBoard** - for monitoring training metrics and visualizing performance.
- **Datasets from Hugging Face** - for loading and processing the NER datasets.
- **Scikit-learn** - used for evaluation metrics such as precision, recall, and F1 score.
- **Optuna** - for hyperparameter optimization to fine-tune model performance.

## Running the Experiments
1. **Prepare the datasets**: Before running the experiments, ensure that the Hindi and Nepali NER datasets are prepared and preprocessed in the correct format. The datasets should follow the CoNLL-2003 standard, with entities labeled appropriately. Place the datasets in the designated directories within the repository.

2. **Training**: To train the models on the Hindi and Nepali datasets, use the `trainer.py` script. This script includes functions for fine-tuning the pre-trained BERT models on the NER tasks. You can adjust hyperparameters such as learning rate, batch size, and number of epochs within the script to suit your specific requirements.

3. **Evaluation**: Once training is complete, the models will be evaluated on monolingual and cross-lingual NER tasks. The evaluation results, including precision, recall, and F1 scores, will be logged using TensorBoard. These metrics can be accessed and visualized to understand the models' performance across different tasks.

## Citation
If you use this repository in your research, please cite it using the following reference:
```
@inproceedings{yadav-etal-2024-cross,
    title = "Cross-Lingual Named Entity Recognition for Low-Resource Languages: A {H}indi-{N}epali Case Study Using Multilingual {BERT} Models",
    author = "Yadav, Dipendra  and
      Suravee, Sumaiya  and
      Strau{\ss}, Tobias  and
      Yordanova, Kristina",
    editor = {S{\"a}lev{\"a}, Jonne  and
      Owodunni, Abraham},
    booktitle = "Proceedings of the Fourth Workshop on Multilingual Representation Learning (MRL 2024)",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.mrl-1.12/",
    doi = "10.18653/v1/2024.mrl-1.12",
    pages = "167--174",
    abstract = "This study investigates the potential of cross-lingual transfer learning for Named Entity Recognition (NER) between Hindi and Nepali, two languages that, despite their linguistic similarities, face significant disparities in available resources. By leveraging multilingual BERT models, including RemBERT, BERT Multilingual, MuRIL, and DistilBERT Multilingual, the research examines whether pre-training them on a resource-rich language like Hindi can enhance NER performance in a resource-constrained language like Nepali and vice versa. The study conducts experiments in both monolingual and cross-lingual settings to evaluate the models' effectiveness in transferring linguistic knowledge between the two languages. The findings reveal that while RemBERT and MuRIL perform well in monolingual contexts{---}RemBERT excelling in Hindi and MuRIL in Nepali{---}BERT Multilingual performs comparatively best in cross-lingual scenarios, in generalizing features across the languages. Although DistilBERT Multilingual demonstrates slightly lower performance in cross-lingual tasks, it balances efficiency with competitive results. The study underscores the importance of model selection based on linguistic and resource-specific contexts, highlighting that general-purpose models like BERT Multilingual are particularly well-suited for cross-lingual applications."
}
```

