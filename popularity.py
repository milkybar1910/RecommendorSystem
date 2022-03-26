class PopularityRecommender:
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(
            items_to_ignore)].sort_values('eventStrength', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]
        return recommendations_df


popularity_model = PopularityRecommender(item_popularity_df, articles_df)
