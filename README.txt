Text Mining for AI — Final Project Code
=======================================

Group: 46
Members: Göktuğ YILDIRIM, Barış KABAN, Murat GÜNDOĞAN, Danila Popuşoi

This zip contains the code and data for the three text mining tasks: sentiment analysis,
named entity recognition (NER), and topic classification. The deep task (2+ systems) is sentiment.


CONTENTS
--------
Notebooks (outputs are already saved inside them):
  Sentiment_task.ipynb   - sentiment: VADER, Naive Bayes (scikit-learn), transformer + variation runs
  NER.ipynb              - NER: Linear SVM on CoNLL-2003 (word + POS features)
  Topic.ipynb            - topic: TF-IDF + Naive Bayes (movie / restaurant / book)

Data:
  Sentiment-topic-test.tsv   - test set (sentence-level sentiment + topic), 10 sentences
  NER-test.tsv               - test set (token-level BIO NER), 214 tokens
  airlinetweets.zip          - sentiment training data (3 classes), used by Sentiment_task.ipynb
  train.txt                  - CoNLL-2003 English training data, used by NER.ipynb

Generated output:
  sentiment_predictions_comparison.csv - per-sentence predictions of all 3 sentiment systems


ENVIRONMENT
-----------
Python 3.11 in a conda environment called tm311.

  conda create -n tm311 python=3.11 -y
  conda activate tm311
  python -m pip install scikit-learn nltk pandas datasets transformers torch

(Optional, only if a notebook asks for it:)
  python -m spacy download en_core_web_sm


HOW TO RUN
----------
1. Keep each notebook in the SAME folder as the data files listed above.
2. Open in VS Code / Jupyter and select the "Python (tm311)" kernel.
3. Run all cells top to bottom.

Notes:
- The notebooks already contain their saved outputs, so re-running is not required.
- NER.ipynb: the SVM training cell takes about 13-14 minutes. This is normal, not a freeze.
- Topic.ipynb and the transformer cell in Sentiment_task.ipynb download models/data on the
  first run, so they need an internet connection the first time.
- Topic.ipynb note: the book-specific reviews dataset would not load with this version of the
  `datasets` library, so the notebook automatically falls back to amazon_polarity (general
  product reviews) as a proxy for the book class. This is printed when it happens.


RESULTS SUMMARY
---------------
Sentiment: VADER macro-F1 0.56 | Naive Bayes 0.48-0.58 | transformer 1.00
NER:       token accuracy 0.94, entity macro-F1 0.54
Topic:     accuracy 0.80, macro-F1 0.81
