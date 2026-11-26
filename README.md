# Fake vs Real News Verifier

A Streamlit app that classifies pasted headlines and articles as Fake or Real using TF-IDF and logistic regression. Its newsroom-style interface shows confidence bars, a suggested news topic, and the words or phrases that influenced the result.

> This academic project looks for writing patterns in its training dataset. It does not independently check sources or verify current events. Model scores are estimates, not proof.

## What the app shows

- Fake and Real model scores as visual bars.
- A suggested topic such as Politics, World News, Business, Health, Technology, Science, Sports, Entertainment, or Environment.
- Key words and short phrases that influenced the Fake/Real prediction.

Topic suggestions use simple keyword matching. They are separate from the trained Fake/Real model.

## How the model works

1. The preparation script combines each article title and body and uses the source file name as its label.
2. The training script splits the data into 80% training and 20% evaluation articles.
3. TF-IDF turns single words and two-word phrases into weighted numbers.
4. Logistic regression learns patterns in those numbers that relate to Fake and Real labels.
5. The script evaluates and saves the model in models/fake_real_model.joblib.

The recorded run reached 99.0% accuracy on a held-back split from this same dataset. That result does not establish accuracy on current or unrelated news.

## Dataset

Download Fake.csv and True.csv from the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). Place them in data/raw/. Dataset files are excluded from this public repository.

## Run locally

From the project folder, install the packages and run:

    python3 -m pip install -r requirements.txt
    python3 src/prepare_data.py
    python3 src/train_model.py
    streamlit run app.py

The app needs models/fake_real_model.joblib. To make a new model, download the two CSV files first and run the preparation and training scripts.
