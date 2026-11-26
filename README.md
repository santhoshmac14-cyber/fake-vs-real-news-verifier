# Fake vs Real News Verifier

A Streamlit app that classifies pasted headlines and articles as Fake or Real using TF-IDF and logistic regression.

## Run in GitHub Codespaces

1. Open this repository on GitHub and choose Code, then Codespaces, then Create codespace on main.
2. Wait for Python and the required packages to install.
3. Choose Terminal, then New Terminal.
4. Run: streamlit run app.py --server.address 0.0.0.0 --server.port 8501
5. Open the forwarded port 8501 link.

Stop the Codespace when you finish. Codespaces uses your account's monthly included usage first.

## About the app

The app uses TF-IDF and logistic regression to estimate whether a pasted headline or article resembles the Fake or Real examples in the training dataset. It shows model confidence scores, a suggested topic, and influential words or phrases. Topic suggestions use simple keyword matching.

This academic project does not independently check sources or verify current events. Scores are estimates, not proof.

## Dataset and model

The model is already included in models/fake_real_model.joblib, so it can run without uploading the large CSV files. Download Fake.csv and True.csv from the Kaggle Fake and Real News Dataset if you want to prepare and train the model again. Dataset files are not included in this public repository.
