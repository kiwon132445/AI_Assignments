from sklearn.feature_selection import RFE, RFECV
from sklearn.decomposition import PCA

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import matplotlib.pyplot as plt

class FeatureAnalysis:
    ###########################################################################################################
    # Kiwon's part
    #
    # Recursive Feature Elimination CV (RFECV)
    def rfecv_init(self, minimum_features=15, step=3, cv=5):
        self.step = step
        self.cv = cv
        self.rfecv = make_pipeline(StandardScaler(), RFECV(estimator=DecisionTreeClassifier(), min_features_to_select=minimum_features, step=step, cv=cv))
        
    def rfecv_fit(self, x, y):
        self.rfecv.fit(x, y)
    
    def rfecv_transform(self, x):
        return self.rfecv.transform(x)
        
    def rfecv_support_(self):
        return self.rfecv[1].support_
    
    def rfecv_score(self, x, y):
        return self.rfecv[1].score(StandardScaler().fit_transform(x), y)
    
    def rfecv_select_plot(self):
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean cross validation score")
        plt.plot(range(1, len(self.rfecv[1].cv_results_["mean_test_score"])*self.step + 1, self.step), self.rfecv[1].cv_results_["mean_test_score"])
        plt.show()
    
    ###########################################################################################################
    # June's part
    #
    # Principal Component Analysis (PCA)
    def pca_init(self, n_components=5):
        self.pca = make_pipeline(StandardScaler(), PCA(n_components=n_components))
        
    def pca_fit_transform(self, x):
        self.pca_ft = self.pca.fit_transform(x)
        return self.pca_ft
    
    def pca_fit_df(self):
        return pd.DataFrame(data=self.pca_ft, columns=self.pca.get_feature_names_out())
    
    ###########################################################################################################
    # John's part
    #
    # Linear Regression (LassoCV)
    def LassoCV_init(self):
        self.LassoCV = LassoCV(cv=KFold(n_splits=5, shuffle=True), max_iter=1000, alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
    
    def LassoCV_fit(self, x, y):
        self.LassoCV.fit(x, y)
    
    def LassoCV_important(self, x, y):
        self.LassoCV_important = [feature for feature, coef in zip(x.columns, self.LassoCV.coef_) if abs(coef) > 0.0001]
        return self.LassoCV_important
