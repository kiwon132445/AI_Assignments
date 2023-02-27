from sklearn.feature_selection import RFE, RFECV
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import matplotlib.pyplot as plt

class FeatureAnalysis:
    ###########################################################################################################
    # Kiwon's part
    #
    #Recursive Feature Elimination CV (RFECV)
    def rfecv_init(self, minimum_features=10, step=5):
        self.step = step
        self.rfecv = make_pipeline(StandardScaler(), RFECV(estimator=LogisticRegression(), step=step, cv=5, min_features_to_select=minimum_features))
        
    def rfecv_fit(self, x, y):
        self.rfecv.fit(x, y)
    
    def rfecv_transform(self, x):
        return self.rfecv.transform(x)
        
    def rfecv_support_(self):
        return self.rfecv.support_
    
    def rfecv_score(self):
        return self.rfecv.score
    
    def rfecv_select_plot(self):
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean cross validation score")
        plt.plot(range(1, len(self.rfecv.cv_results_["mean_test_score"])*self.step + 1, self.step), self.rfecv.cv_results_["mean_test_score"])
        plt.show()
    
    ###########################################################################################################
    # June's part
    #
    #Principal Component Analysis
    def pca_init(self, n_components=5):
        self.pca = make_pipeline(StandardScaler(), PCA(n_components=n_components))
        
    def pca_fit_transform(self, x):
        self.pca_ft = self.pca.fit_transform(x)
        return self.pca_ft
    
    def pca_fit_df(self):
        return pd.DataFrame(data=self.pca_ft, columns=self.pca.get_feature_names_out())
    