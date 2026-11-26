# Fake vs Real News Verifier

A Streamlit web app that classifies a pasted news headline or article as **Fake** or **Real** using TF-IDF and logistic regression. It also shows the model's class scores and the words or phrases that contributed most to its prediction.

> This is a text-pattern classifier for an academic project. It does not independently fact-check current events. Its scores are model estimates, not proof that a story is true or false.

## What the project uses

- **Python** for the app and training code.
- **Python's built-in csv module** to read and prepare the data.
- **scikit-learn** for the train/test split, TF-IDF, logistic regression, and evaluation.
- **Streamlit** for the paste-text web page.
- **joblib** to save and load the trained model.

## Data

The project uses Kaggle's Fake and Real News Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset. Put Fake.csv and True.csv in:

    data/raw/Fake.csv
    data/raw/True.csv

Rows from Fake.csv get the label Fake; rows from True.csv get the label Real. The preparation script combines the title and article body. If a body is blank, the title is still used. Subject and date are not model inputs. The dataset files are not included in this public repository; .gitignore keeps the local data/ folder out.

## How the model works

1. src/prepare_data.py reads the two source CSV files and writes text/label pairs to data/processed/news_labeled.csv.
2. src/train_model.py sets aside 20% of the articles for evaluation, keeping the Fake/Real balance similar in both groups.
3. TfidfVectorizer converts training text into weighted numbers for single words and two-word phrases (unigrams and bigrams). It ignores terms seen in fewer than two training articles and caps the vocabulary at 50,000 features.
4. Logistic regression learns how those numbers relate to Fake and Real labels.
5. The script reports performance on the held-back articles and saves the pipeline to models/fake_real_model.joblib.
6. app.py loads that model, predicts a pasted text, and displays each class score.

The phrases shown in the app are found by multiplying a phrase's TF-IDF value in the pasted text by logistic regression's learned weight, then sorting the strongest contributions for the predicted class. This explains model behavior; it is not evidence about the story.

## Evaluation on this dataset split

The recorded run used a stratified 80/20 split and reached **99.0% accuracy** on the 8,980 held-back articles. Precision, recall, and F1 were each about 0.99 for both classes on that split. This measures performance on held-back examples from the same dataset; it does not establish accuracy on current or unrelated news.

## Run it locally

Use Python 3.9 or newer. From the project folder, install the listed packages and run:

    python3 -m pip install -r requirements.txt
    python3 src/prepare_data.py
    python3 src/train_model.py
    streamlit run app.py

If you already have the saved model file, the app can start with streamlit run app.py. For training from scratch, first put both Kaggle CSVs in data/raw/ and run the preparation and training commands.

## Files

- app.py — Streamlit interface for pasted text, class scores, and key n-grams.
- src/prepare_data.py — reads the two CSVs, combines title and body, and assigns labels.
- src/train_model.py — splits the data, trains and evaluates the model, then saves it.
- models/fake_real_model.joblib — saved TF-IDF and logistic regression pipeline used by the app.
- requirements.txt — Python packages needed by the project.
