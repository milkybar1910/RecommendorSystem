import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:

    def __init__(self, item_ids, tfidf_matrix, user_profiles, items_df=None):
        self.item_ids = item_ids
        self.tfidf_matrix = tfidf_matrix
        self.user_profiles = user_profiles
        self.items_df = items_df

    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        cosine_similarities = cosine_similarity(
            self.user_profiles[person_id], self.tfidf_matrix)
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0, i])
                               for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(
            user_id)
        similar_items_filtered = list(
            filter(lambda x: x[0] not in items_to_ignore, similar_items))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=[
                                          'contentId', 'recStrength']).head(topn)
        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]
        return recommendations_df
