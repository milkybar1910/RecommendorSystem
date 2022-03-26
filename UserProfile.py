import sklearn
import numpy as np
import scipy


class User:

    def __init__(self, item_ids, articles_df, interactions_train_df, tfidf_matrix):
        self.item_ids = item_ids
        self.interactions_train_df = interactions_train_df
        self.articles_df = articles_df
        self.tfidf_matrix = tfidf_matrix

    def get_item_profile(self, item_id):
        idx = self.item_ids.index(item_id)
        item_profile = self.tfidf_matrix[idx:idx+1]
        return item_profile

    def get_item_profiles(self, ids):
        item_profiles_list = [self.get_item_profile(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_users_profile(self, person_id, interactions_indexed_df):
        interactions_person_df = interactions_indexed_df.loc[person_id]
        user_item_profiles = self.get_item_profiles(
            interactions_person_df['contentId'])

        user_item_strengths = np.array(
            interactions_person_df['eventStrength']).reshape(-1, 1)
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(
            user_item_strengths), axis=0) / np.sum(user_item_strengths)
        user_profile_norm = sklearn.preprocessing.normalize(
            user_item_strengths_weighted_avg)
        return user_profile_norm

    def build_users_profiles(self):
        interactions_indexed_df = self.interactions_train_df[self.interactions_train_df['contentId'].isin(
            self.articles_df['contentId'])].set_index('personId')
        user_profiles = {}
        for person_id in interactions_indexed_df.index.unique():
            user_profiles[person_id] = self.build_users_profile(
                person_id, interactions_indexed_df)
        return user_profiles
