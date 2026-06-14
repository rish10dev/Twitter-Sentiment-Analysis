# Twitter Sentiment Analysis

## Dataset

This project uses the Sentiment140 dataset.

The dataset is not included in this repository because it exceeds GitHub's file size limit (100 MB).

Download the dataset from:

https://www.kaggle.com/datasets/kazanova/sentiment140

After downloading, create the following folder structure:

Twitter-Sentiment-Analysis/
├── data/
│   └── training.1600000.processed.noemoticon.csv

Then run:

python src/sentiment_analysis.py

## Results

* Accuracy: 74.53%
* Algorithm: Logistic Regression
* Feature Extraction: TF-IDF Vectorization

The trained model and vectorizer are automatically saved in the models folder after execution.
