import pickle

import pandas as pd
import re
from textblob import TextBlob

import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
import sklearn

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":

        inp = request.form.get("inp")

        vocab = pickle.load(open("../nlp/01-machine_learning/bow.pkl", "rb"))
        classifier = pickle.load(open("../nlp/01-machine_learning/model.pkl", "rb"))

        def preprocess_str(text):

            flag = 'stem'

            text = re.sub("n't", " not", text)
            text = re.sub("\:\(", "bad", text)
            text = re.sub("\:\)", "good", text)
            text = re.sub("not good", "bad", text)
            text = re.sub("not great", "bad", text)
            text = re.sub("not bad", "average", text)

            # 1. Removing special characters and digits
            sentence = re.sub("[^a-zA-Z]", " ", text)

            # 2. Converting to lowercase
            sentence_1 = sentence.lower()

            # 3. Tokenization (Word-level)
            tokens = sentence_1.split()

            # 3.1. TextBlob- Correcting the spellings
            tokens_correct_spell = [str(TextBlob(token)) for token in tokens]

            # 4. Removing stopwords
            extra_stops = ['br', 'http', 'www', 'k']
            stop_words = [x for x in stopwords.words('english') if x not in ['not', 'but']] + extra_stops

            clean_tokens = [token for token in tokens_correct_spell if token not in stop_words]

            # 5. stemming
            if flag == 'stem':
                clean_tokens_final = [stemmer.stem(word) for word in clean_tokens]
            else:
                clean_tokens_final = [lemmatizer.lemmatize(word) for word in clean_tokens]

            return pd.Series([" ".join(clean_tokens_final)])

        transformed_text = vocab.transform(preprocess_str(inp))
        prediction = classifier.predict(transformed_text)[0]

        if prediction == "Positive":
            return render_template("predict.html", message = "Positive")
        elif prediction == "Negative":
            return render_template("predict.html", message = "Negative")
        else:
            pass

    return render_template('home.html')


if __name__ == '__main__':
    app.run()