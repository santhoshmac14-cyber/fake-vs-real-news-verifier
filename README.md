# Fake vs Real News Verifier

A small browser app that classifies pasted news text with the project's trained TF-IDF and logistic regression model. It displays Fake/Real model scores, a suggested topic, and words or two-word phrases that influenced the result.

> This classroom model recognizes patterns in the dataset it learned from. It does not check sources or independently verify events. Scores are estimates, not proof.

## Open the live app

The website is hosted with GitHub Pages: https://santhoshmac14-cyber.github.io/fake-vs-real-news-verifier/

## Open and run the project in VS Code

1. Install VS Code and the official GitHub Codespaces extension, then sign in to GitHub.
2. Open this repository on GitHub. Choose **Code** → **Codespaces** → **Create codespace on main**.
3. When the Codespace opens in your browser, use its menu to choose **Open in VS Code**. Approve the prompt to open VS Code if it appears.
4. In VS Code, choose **Terminal** → **New Terminal**.
5. Start the local website with:

       python3 -m http.server 8000 --directory docs

6. Open the forwarded port **8000** to use the app. Stop the server with **Control+C** when finished.

The Codespace is a cloud computer. GitHub includes a monthly usage allowance; stop the Codespace when you finish.

## Open and run locally in VS Code

1. Download this repository from GitHub using **Code** → **Download ZIP**, then unzip it.
2. In VS Code, choose **File** → **Open Folder** and select the unzipped project folder.
3. Open **Terminal** → **New Terminal** and run:

       python3 -m http.server 8000 --directory docs

4. In a browser, open http://localhost:8000.

Python's built-in web server just serves the website files. The classifier itself runs in the browser.

## How it works

- `docs/index.html` contains the page structure.
- `docs/styles.css` contains the visual design.
- `docs/scripts/app.js` performs TF-IDF calculation, logistic regression scoring, topic suggestion, and key phrase display in the browser.
- `docs/model.json` contains the exported vocabulary and trained model weights. It is derived from `models/fake_real_model.joblib` and is used directly by the browser.
- `src/` contains the basic-data preparation, splitting, model training, and model export scripts.

The TF-IDF vectorizer uses single words and two-word phrases, English stop words, and L2 normalization. The logistic regression model was trained on the Fake and Real news dataset. The existing held-out accuracy result reflects a split from that same dataset; it does not guarantee performance on new/current stories.

## Retrain the model

Download `Fake.csv` and `True.csv` from the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) into `data/raw/`. Then follow the data preparation and training steps in `src/`. After training, create the browser model file with:

       python3 src/export_model.py

Do not upload the original CSV dataset files to the public repository.
