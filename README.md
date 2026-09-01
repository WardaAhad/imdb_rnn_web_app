# IMDb RNN Genre Classifier — Backend + Frontend

This project wraps the trained model from `IMDB_RNN_Project.ipynb` in a FastAPI backend and Streamlit frontend.

## Model used

The notebook uses:

- `Tokenizer(num_words=5000)`
- `pad_sequences(..., maxlen=200)`
- `Embedding(input_dim=5000, output_dim=64)`
- `SimpleRNN(64)`
- `Dense(32, relu)`
- `Dense(202, softmax)`

The notebook trains for 5 epochs with batch size 32 and reports test accuracy of about 7.5%. The target is **movie genre**, not positive/negative sentiment.

## Project structure

```text
imdb_rnn_web_app/
├── backend/
│   ├── main.py
│   ├── prepare_artifacts.py
│   ├── requirements.txt
│   ├── tokenizer.pkl          # generated
│   └── label_encoder.pkl      # generated
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── model/
│   └── imdb_rnn_model.keras
├── imdb_dataset.csv           # place your original dataset here
└── README.md
```

## Important

The `.keras` model does **not** contain the Keras `Tokenizer` or sklearn `LabelEncoder` from the notebook. Therefore, generate those artifacts from the same `imdb_dataset.csv` used in training.

The preprocessing in `prepare_artifacts.py` matches the notebook:
- lowercase text
- remove non A-Z characters
- `Tokenizer(num_words=5000)`
- fit tokenizer on `Overview`
- fit `LabelEncoder` on `Genre`

## Run backend

From the project root:

```bash
pip install -r backend/requirements.txt
python backend/prepare_artifacts.py
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Run frontend

Open a second terminal:

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

The Streamlit app sends the overview to:

```text
POST http://127.0.0.1:8000/predict
```

## API example

Request:

```json
{
  "overview": "A detective investigates a mysterious murder in a small town."
}
```

Response:

```json
{
  "genre": "Crime, Drama",
  "confidence": 0.12,
  "top_predictions": [
    {
      "genre": "Crime, Drama",
      "confidence": 0.12
    }
  ]
}
```

The exact predicted values depend on the trained model and input text.
