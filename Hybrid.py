class HybridRecommender:

    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how='outer',
                                   left_on='contentId',
                                   right_on='contentId').fillna(0.0)

        recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) + (
            recs_df['recStrengthCF'] * self.cf_ensemble_weight)

        recommendations_df = recs_df.sort_values(
            'recStrengthHybrid', ascending=False).head(topn)
        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[['recStrengthHybrid', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df
