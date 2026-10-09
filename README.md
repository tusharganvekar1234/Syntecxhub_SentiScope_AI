
# SentiScope AI

An interactive sentiment analysis application built with Python, NLP and Machine Learning.

## Features
- TF-IDF text feature extraction
- Logistic Regression sentiment classifier
- Accuracy and macro F1 evaluation
- Interactive sentiment prediction
- Batch CSV review analysis
- Sentiment visualizations using Plotly
- Downloadable analysis results

## Dataset
Uses the UCI Sentiment Labelled Sentences dataset containing 3,000 labeled sentences from Amazon, IMDb and Yelp.

Dataset labels:
- 0 = Negative
- 1 = Positive

## Model
- Text representation: TF-IDF with unigrams and bigrams
- Classifier: Logistic Regression
- Split: 80% training / 20% testing
- Random state: 42

## Evaluation
- Accuracy: 81.3%
- Macro F1-score: 0.813

These results come from the current held-out test split. Results may vary if the data or configuration changes.

## Run Locally

```bash
pip install -r requirements.txt
python prepare_data.py
python train_model.py
streamlit run app.py
```

## Project Structure

```text
Syntecxhub_SentiScope_AI/
├── app.py
├── train_model.py
├── prepare_data.py
├── requirements.txt
├── README.md
└── data/
    ├── amazon_cells_labelled.txt
    ├── imdb_labelled.txt
    ├── yelp_labelled.txt
    └── reviews.csv
```

## Limitations
This is an educational project trained on a small labeled sentence dataset. It predicts only positive or negative sentiment and may struggle with sarcasm, mixed opinions and unfamiliar domains.

## Built With
Python, pandas, scikit-learn, Streamlit and Plotly.
