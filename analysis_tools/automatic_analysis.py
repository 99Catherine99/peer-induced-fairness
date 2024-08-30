from preprocessing.encoding import FeatureEncoder
from model.logistic import LogisticRegressionModel
from model.rf import RFModel
from model.xgboost import XGBoostModel
from matching.caplier_matching import CaplierMatching
from analysis.bootstrapping import BootstrapSampler
from analysis import data_divide_calculator



class AutomatedAnalysis:
    def __init__(self, df, nominals, columns_to_drop, drop_list, group_col, target_col_1, target_col_2, K, caplier_ratio, alpha, model_type_1='LR', model_type_2='LR'):
        self.df = df
        self.nominals = nominals
        self.drop_list = drop_list
        self.group_col = group_col
        self.target_col_1 = target_col_1
        self.target_col_2 = target_col_2
        self.columns_to_drop = columns_to_drop
        self.K = K
        self.caplier_ratio = caplier_ratio
        self.alpha = alpha
        self.model_type_1 = model_type_1
        self.model_type_2 = model_type_2

    def preprocess_and_encode(self):
        self.df = self.df.drop(columns=self.columns_to_drop)
        X_part = self.df.drop(self.drop_list, axis=1)
        X_part_nominals = [x for x in X_part if x in self.nominals]
        encoder = FeatureEncoder(X_part)
        self.df_encoded = encoder.one_hot_encode(columns=X_part_nominals)
        return self.df_encoded

    def train_and_predict(self):

        if self.model_type_1 == 'RF':
            model_1 = RFModel(self.df_encoded, self.df[self.target_col_1])
        elif self.model_type_1 == 'LR':
            model_1 = LogisticRegressionModel(self.df_encoded, self.df[self.target_col_1])
        elif self.model_type_1 == 'XGB':
            model_1 = XGBoostModel(self.df_encoded, self.df[self.target_col_1])
        else:
            raise ValueError(f"Unsupported model type {self.model_type_1}")


        model_1.train_model()
        model_1.evaluate_performance()
        model_1.generate_test_report()
        model_1.calculate_auc()
        model_1.calculate_acc()
        model_1.calculate_f1()
        model_1.calculate_recall()
        model_1.calculate_precision()

        yb_pred_1 = model_1.predict_proba(self.df_encoded)
        yb_prediction_1 = model_1.predict(self.df_encoded)

        merge1_df = self.df.copy()
        merge1_df['Pr(S=0)'] = yb_pred_1[:, 0]
        merge1_df['PS Prediction'] = yb_prediction_1

        if self.model_type_2 == 'RF':
            model_2 = RFModel(self.df_encoded, self.df[self.target_col_2])
        elif self.model_type_2 == 'LR':
            model_2 = LogisticRegressionModel(self.df_encoded, self.df[self.target_col_2])
        elif self.model_type_2 == 'XGB':
            model_2 = XGBoostModel(self.df_encoded, self.df[self.target_col_2])
        else:
            raise ValueError(f"Unsupported model type {self.model_type_2}")

        model_2.train_model()
        model_2.evaluate_performance()
        model_2.generate_test_report()
        model_2.calculate_auc()
        model_2.calculate_acc()
        model_2.calculate_f1()
        model_2.calculate_recall()
        model_2.calculate_precision()

        yb_pred_2 = model_2.predict_proba(self.df_encoded)
        yb_prediction_2 = model_2.predict(self.df_encoded)

        self.merge_pred_ps_df = merge1_df.copy()
        self.merge_pred_ps_df['Pr(Y=1)'] = yb_pred_2[:, 1]
        self.merge_pred_ps_df['Binary Prediction'] = yb_prediction_2

        return self.merge_pred_ps_df


    def perform_matching_and_testing(self):
        divider_results = data_divide_calculator.data_divide(self.merge_pred_ps_df, self.group_col, 'Binary Y', 'Binary Prediction', 'Pr(S=0)')
        self.protect_df, self.nonprotect_df, self.accepted_df, self.rejected_df, self.pred_accepted_df, self.pred_rejected_df, self.protect_ps, self.nonprotect_ps = divider_results

        matcher = CaplierMatching(self.merge_pred_ps_df, self.group_col, self.K, self.caplier_ratio)
        self.matched_df = matcher.caplier_matching()

        sampler = BootstrapSampler(self.nonprotect_df, self.protect_df, self.matched_df, sampling_times=100, draws_per_sample=30,
                                   determine=35)
        self.bootstrapped_samples = sampler.sample()
        self.pr_y_means = sampler.calculate_means(self.bootstrapped_samples)

        self.single_less_sided_results = sampler.perform_test(self.pr_y_means, direction='less')
        self.single_greater_sided_results = sampler.perform_test(self.pr_y_means, direction='greater')
        self.double_sided_results = sampler.perform_test(self.pr_y_means, direction='two-sided')

        self.plot = sampler.plot_treatment_comparison(self.pr_y_means,self.double_sided_results)


        return self.matched_df, self.bootstrapped_samples, self.pr_y_means, self.single_less_sided_results, self.single_less_sided_results, self.double_sided_results



    def full_analysis(self):
        self.preprocess_and_encode()
        self.train_and_predict()
        self.perform_matching_and_testing()


        return {
            'merge_pred_ps_df': self.merge_pred_ps_df,
            'matched_df': self.matched_df,
            'bootstrapped_samples': self.bootstrapped_samples,
            'pr_y_means': self.pr_y_means,
            'single_greater_sided_results': self.single_greater_sided_results,
            'single_less_sided_results': self.single_less_sided_results,
            'double_sided_results': self.double_sided_results
        }



