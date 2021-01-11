from pydoc import html

from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#import csv file
data = pd.read_csv("cleaned_review_product.csv")

pkl_transform_filename = "transform.pkl"
pkl_classifier_filename = "log_classifier.pkl"
# Load the Model back from file
tf = pickle.load(open(pkl_transform_filename, 'rb'))
clf = pickle.load(open(pkl_classifier_filename, 'rb'))

# Create the application.
app = Flask(__name__)

@app.route('/')
def index():
    recommend_final = recommend()
    return render_template('index.html', recommend_final = [recommend_final.to_html()],titles = ['Product', 'Recommended Products'] )

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        reviewText = request.form['reviewText']
        reviewTextCV = tf.transform([reviewText])
        reviewTextClf = clf.predict(reviewTextCV)
        final_prediction = 'Negative' if reviewTextClf[0] == 0 else 'Positive'

    return render_template('result.html', prediction = final_prediction)

@app.route('/summaryreview', methods=['POST'])

def summaryreview():
    if request.method == 'POST':
        # Classify ratings as good, neutral and bad reviews
        reviewSummary = ""

    return render_template('summary.html',  reviewSummary = getSummary())

def getSummary():
    total_reviews = len(data['clean_text'])
    good_rate = len(data[data['reviews.rating'] > 3])
    neut_rate = len(data[data['reviews.rating'] == 3])
    bad_rate = len(data[data['reviews.rating'] < 3])

    ratingArray = [total_reviews,good_rate,neut_rate,bad_rate]
    return ratingArray


@app.route('/features', methods=['POST'])

def features():

    word_features = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', stop_words='english', ngram_range=(1, 2), max_features=None)
    vectors = word_features.fit_transform(data['clean_text'])
    feature_names = word_features.get_feature_names()
    features =  top_feats_in_doc(vectors, feature_names, 0 , 25)
    features_final = features

    return render_template('features.html', features_final = [features_final.to_html()],titles = ['Index', 'Feature', 'Weightage'] )


def top_tfidf_feats(row, features, top_n=25):
     # Get top n tfidf values in row and return them with their corresponding feature names.'''
     topn_ids = np.argsort(row)[::-1][:top_n]
     top_feats = [(features[i], row[i]) for i in topn_ids]
     top_feats_df = pd.DataFrame(top_feats)
     top_feats_df.columns = ['feature', 'tfidf']
     return top_feats_df


def top_feats_in_doc(vectors, features, row_id, top_n=25):
    #Top tfidf features in specific document (matrix row)
    row = np.squeeze(vectors[row_id].toarray())

    return top_tfidf_feats(row, features, top_n)


def recommend():
    recomm_df = data[['product', 'reviews.rating']]
    # Getting the average rating product
    avg_rating_prod = recomm_df.groupby('product').sum() / recomm_df.groupby('product').count()
    avg_rating_prod = avg_rating_prod.reset_index()

    # Top 10 Highly rated products(Popularity based)
    return avg_rating_prod.nlargest(10, 'reviews.rating')

if __name__ == '__main__':
	app.run(debug=True)