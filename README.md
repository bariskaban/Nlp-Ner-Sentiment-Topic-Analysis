# Text Mining for AI 
**Authors:** Goktu YILDIRIM, Baris KABAN, Murat GUNDOGAN, Danila Popusoi  

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Tasks](#tasks)
  - [Sentiment Analysis (Deep Task)](#sentiment-analysis-deep-task)
  - [Named Entity Recognition](#named-entity-recognition)
  - [Topic Classification](#topic-classification)
- [Data](#data)
- [Data Acquisition](#data-acquisition)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)

---

## Project Overview

This project implements and evaluates three text mining tasks: sentiment analysis, named entity recognition (NER), and topic classification. Sentiment analysis is designated as the deep task and is approached with multiple systems of increasing complexity, from a rule-based lexicon method to a fine-tuned transformer model.

All tasks are evaluated against a shared held-out test set of 10 sentences with gold-standard annotations for sentiment, topic, and named entities.

---

## Repository Structure

```
Group46_project/
|-- Sentiment_task.ipynb                  # Sentiment analysis (deep task): VADER, Naive Bayes, Transformer
|-- NER.ipynb                             # Named entity recognition with a linear SVM
|-- Topic.ipynb                           # Topic classification with TF-IDF and Multinomial Naive Bayes
|-- Sentiment-topic-test.tsv              # Shared test set: 10 sentences with sentiment and topic labels
|-- NER-test.tsv                          # NER test set: 214 tokens with BIO annotations
|-- train.txt                             # CoNLL-2003 training corpus for NER (203,621 tokens)
|-- sentiment_predictions_comparison.csv  # Per-sentence comparison of all three sentiment systems
└── airlinetweets/                        # Sentiment training corpus (4,755 tweets)
|   |-- positive/                         # 1,490 positive tweets
|   |-- negative/                         # 1,750 negative tweets
|   └── neutral/                          # 1,515 neutral tweets
```

---

## Tasks

### Sentiment Analysis (Deep Task)

**Notebook:** [Sentiment_task.ipynb](Sentiment_task.ipynb)

The sentiment task classifies text into three categories: positive, negative, and neutral. Three distinct systems are implemented and compared.

**Training Data:** The airline tweets corpus (`airlinetweets/`) contains 4,755 labeled tweets sourced from airline-related discussions on Twitter, split across positive (1,490), negative (1,750), and neutral (1,515) classes.

#### System 1: VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based, lexicon-driven sentiment analyzer from the NLTK library. It requires no training and operates by assigning valence scores to individual words and aggregating them into a compound score. Classification thresholds are set at >= 0.05 for positive, <= -0.05 for negative, and the range in between for neutral.

- **Macro-F1 on test set:** 0.56

#### System 2: Naive Bayes

A Multinomial Naive Bayes classifier is trained on the airline tweets corpus using NLTK for tokenization and stopword removal. Multiple hyperparameter configurations are evaluated, varying TF-IDF weighting (enabled or disabled) and minimum document frequency thresholds (2, 5, and 10).

- **Best macro-F1 on test set:** 0.58 (no TF-IDF, `min_df` = 5 or 10)
- **Default macro-F1 on test set:** 0.48 (TF-IDF enabled, `min_df` = 2)

#### System 3: Transformer

A pre-trained transformer model is applied via the HuggingFace `transformers` library. The primary model used is `cardiffnlp/twitter-roberta-base-sentiment-latest`, which is specifically fine-tuned for three-class sentiment on Twitter data and directly matches the domain of the training corpus. A secondary model, `distilbert-base-uncased-finetuned-sst-2-english`, is also tested with a confidence-based neutral band applied to its binary output.

- **Macro-F1 on test set:** 1.00

---

### Named Entity Recognition

**Notebook:** [NER.ipynb](NER.ipynb)

The NER task identifies and classifies named entities in text using the standard BIO tagging scheme. Entity types include PER (person), ORG (organization), LOC (location), and MISC (miscellaneous).

**Training Data:** The CoNLL-2003 English corpus ([train.txt](train.txt)), which contains 203,621 tokens with POS tags and BIO NER annotations. The corpus is heavily skewed toward outside tokens (O), which account for approximately 83% of all tokens.

**System:** A Linear SVM classifier trained on word and POS tag pairs extracted using NLTK's averaged perceptron tagger. Features are encoded using `DictVectorizer` from scikit-learn. Training takes approximately 13 to 14 minutes due to corpus size.

**Results on 214 test tokens:**

| Metric | Value |
|---|---|
| Token accuracy | 0.94 |
| Entity macro-F1 | 0.54 |

Notable per-class F1 scores: I-PER (0.74), I-LOC (0.80), B-MISC (0.67), B-LOC (0.50), B-ORG (0.00). The classifier struggles most with organizational entities, frequently confusing them with person or location tags. Representative errors include "Warner Brothers" predicted as I-PER and "New York University" predicted as a location sequence.

---

### Topic Classification

**Notebook:** [Topic.ipynb](Topic.ipynb)

The topic task classifies short texts into one of three domains: movie review, restaurant review, or book review.

**Training Data:** 6,000 reviews (2,000 per class) drawn from three sources. Movie reviews come from the Rotten Tomatoes dataset. Restaurant reviews come from the Yelp dataset. Book reviews use the Amazon Polarity dataset as a proxy due to version incompatibility with the original Amazon book review corpus.

**System:** TF-IDF vectorization (`min_df` = 2, English stopwords removed) combined with a Multinomial Naive Bayes classifier.

**Results on 10 test sentences:**

| Metric | Value |
|---|---|
| Accuracy | 0.80 |
| Macro-F1 | 0.81 |

Per-class: restaurant (F1 = 1.00), movie (F1 = 0.75), book (F1 = 0.67). The two misclassified sentences are movie reviews incorrectly predicted as book reviews, likely attributable to domain shift from using the Amazon proxy dataset for book training data.

---

## Data

### Sentiment-topic-test.tsv

Tab-separated file with columns: `sentence_id`, `text`, `sentiment`, `topic`. Contains 10 annotated test sentences used for evaluating both the sentiment and topic tasks. Sentiment labels are negative, neutral, and positive. Topic labels are movie, restaurant, and book.

### NER-test.tsv

Tab-separated file with columns: `sentence_id`, `token_id`, `token`, `NER_tag`. Contains 214 tokens from 10 sentences with gold BIO NER annotations.

### train.txt

CoNLL-2003 format. Each line contains a word followed by its POS tag, chunk tag, and NER tag separated by spaces. Blank lines separate sentences and lines beginning with `-DOCSTART-` mark document boundaries.

### airlinetweets/

One plain-text file per tweet, named `AL_<tweet_id>.txt`, organized into `positive/`, `negative/`, and `neutral/` subdirectories. Used exclusively for training the Naive Bayes sentiment classifier.

---

## Data Acquisition

Two datasets are excluded from this repository due to licensing restrictions and must be obtained separately before running the notebooks.

**Airline tweets (`airlinetweets/`)**

Twitter's Terms of Service prohibit redistribution of tweet content. The dataset used here is the Kaggle "Twitter US Airline Sentiment" dataset, originally compiled by Crowdflower. Download it from Kaggle and extract it so that the `airlinetweets/` directory contains three subdirectories -- `positive/`, `negative/`, and `neutral/` -- each holding one `.txt` file per tweet named `AL_<tweet_id>.txt`.

**CoNLL-2003 NER corpus (`train.txt`)**

The CoNLL-2003 English NER dataset is derived from the Reuters Corpus Volume 1 (RCV1), which requires a separate license agreement. The data can be obtained through the Linguistic Data Consortium (LDC) or through your institution's licensed access. The expected format is one token per line with space-separated columns: `word POS-tag chunk-tag NER-tag`, with blank lines between sentences and `-DOCSTART-` markers between documents.

---

## Environment Setup

Python 3.11 is required. The recommended setup uses a conda environment.

```bash
conda create -n tm311 python=3.11 -y
conda activate tm311
python -m pip install scikit-learn nltk pandas datasets transformers torch
```

To optionally install spaCy (not required for any notebook):

```bash
python -m spacy download en_core_web_sm
```

NLTK data packages are downloaded automatically within the notebooks on first run. The transformer and HuggingFace datasets cells require an internet connection on first execution to download model weights and datasets.

---

## How to Run

1. Keep all notebooks in the same directory as the data files. The notebooks use relative paths.
2. Open the project folder in VS Code or Jupyter and select the `tm311` kernel (Python 3.11).
3. Run all cells in the desired notebook from top to bottom.

**Important notes:**

- All notebooks already contain saved cell outputs. Re-execution is optional for reviewing results.
- `NER.ipynb` SVM training is expected to take 13 to 14 minutes on first run. This is normal behavior and does not indicate a problem.
- `Topic.ipynb` and the transformer cells in `Sentiment_task.ipynb` download external data and model weights on first run and require an active internet connection.

---

## Results Summary

| Task | System | Metric | Score |
|---|---|---|---|
| Sentiment | Transformer (twitter-roberta-base) | Macro-F1 | 1.00 |
| Sentiment | VADER | Macro-F1 | 0.56 |
| Sentiment | Naive Bayes (best config) | Macro-F1 | 0.58 |
| NER | Linear SVM | Token Accuracy | 0.94 |
| NER | Linear SVM | Entity Macro-F1 | 0.54 |
| Topic | TF-IDF + Naive Bayes | Accuracy | 0.80 |
| Topic | TF-IDF + Naive Bayes | Macro-F1 | 0.81 |

Per-sentence predictions for the sentiment task across all three systems are recorded in [sentiment_predictions_comparison.csv](sentiment_predictions_comparison.csv).
