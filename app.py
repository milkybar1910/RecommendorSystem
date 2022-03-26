
import pickle
from flask import Flask, render_template, request


app = Flask(__name__)

hybrid_recommender_model = pickle.load(
    open("hybrid_recommender_model.pkl", "rb"))


@app.route('/predict/', methods=['GET'])
def predict():
    # n = request.args.get("id")
    n = -1479311724257856983
    # print()
    print("Siva")
    hybrid = hybrid_recommender_model.recommend_items(
        n, topn=20, verbose=True)
    dict = {
        "recommendation": list(hybrid["contentId"])
    }
    return dict


@app.route('/')
def home():
    return "siva"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
