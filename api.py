from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
from flask_cors import CORS

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
        scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
        cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

        # Check if file or text
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions_csv, graph = bulk_prediction(predictor, scaler, cv, data)

            # Convert CSV to base64
            csv_base64 = base64.b64encode(predictions_csv.getvalue()).decode('ascii')
            
            # Convert graph to base64
            graph_base64 = base64.b64encode(graph.getvalue()).decode('ascii')

            # Return JSON response with both CSV and graph data
            return jsonify({
                "success": True,
                "csv_data": csv_base64,
                "graph_data": graph_base64,
                "filename": "Predictions.csv"
            })

        else:
            # Handle JSON text input
            data = request.get_json()
            if data and "text" in data:
                text_input = data["text"]
                predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
                return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"


def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)
    return predictions_csv, graph


def get_distribution_graph(data):
    """Create pie chart of sentiment distribution"""
    fig = plt.figure(figsize=(5, 5))
    color_map = {"Positive": "green", "Negative": "red", "Neutral": "blue"}
    wp = {"linewidth": 1, "edgecolor": "black"}

    # Use correct column name
    tags = data["Predicted sentiment"].value_counts()

    # Dynamically handle explode length
    explode = tuple([0.01] * len(tags))

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=[color_map.get(label, "gray") for label in tags.index],
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        ylabel=""
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()
    graph.seek(0)
    return graph


def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)