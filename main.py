
import math
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from pymongo import MongoClient
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse.linalg import svds


from contentBased import ContentBasedRecommender
from UserProfile import User
from Collabration import CFRecommender
from Hybrid import HybridRecommender

connection = MongoClient('localhost', 27017)
db = connection.placement
shared = db.shared

articles_df = pd.DataFrame(list(shared.find()))
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']

interactions = db.interactions
interactions_df = pd.DataFrame(list(interactions.find()))

event_type_strength = {
    'VIEW': 1.0,
    'LIKE': 2.0,
    'BOOKMARK': 2.5,
    'FOLLOW': 3.0,
    'COMMENT CREATED': 4.0,
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(
    lambda x: event_type_strength[x])
users_interactions_count_df = interactions_df.groupby(
    ['personId', 'contentId']).size().groupby('personId').size()
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[
    ['personId']]

interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df,
                                                            how='right',
                                                            left_on='personId',
                                                            right_on='personId')


def smooth_user_preference(x):
    return math.log(1+x, 2)


interactions_full_df = interactions_from_selected_users_df \
    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
    .apply(smooth_user_preference).reset_index()

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                               stratify=interactions_full_df['personId'],
                                                               test_size=0.20,
                                                               random_state=42)


vectorizer = TfidfVectorizer(analyzer='word',
                             ngram_range=(1, 2),
                             min_df=0.003,
                             max_df=0.5,
                             max_features=5000,
                             stop_words=stopwords.words('english'))

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(
    articles_df['title'] + "" + articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()


user = User(item_ids, articles_df, interactions_train_df, tfidf_matrix)

user_profiles = user.build_users_profiles()

content_based_recommender_model = ContentBasedRecommender(
    item_ids, tfidf_matrix, user_profiles, articles_df)

users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId',
                                                          columns='contentId',
                                                          values='eventStrength').fillna(0)
users_items_pivot_matrix = users_items_pivot_matrix_df.values
users_ids = list(users_items_pivot_matrix_df.index)
users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

NUMBER_OF_FACTORS_MF = 15
U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=NUMBER_OF_FACTORS_MF)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
    all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm,
                           columns=users_items_pivot_matrix_df.columns, index=users_ids).transpose()


cf_recommender_model = CFRecommender(cf_preds_df, articles_df)


hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df,
                                             cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)

pickle.dump(hybrid_recommender_model, open(
    "hybrid_recommender_model.pkl", "wb"))
