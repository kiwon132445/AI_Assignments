from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt

class FeatureAnalysis:
    def __init__ (self, minimum_features=10, step=5):
        self.step = step
        self.rfecv = RFECV(estimator=LogisticRegression(), step=step, cv=5, min_features_to_select=minimum_features)
    def rfecv_fit(self, x, y):
        self.rfecv.fit(x, y)
        
    def rfecv_x(self, x):
        return x[x.columns[self.rfecv.support_]]
    
    def rfecv_score(self):
        return self.rfecv.score
    
    def rfecv_plotting(self):
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean cross validation score")
        plt.plot(range(1, len(self.rfecv.cv_results_["mean_test_score"])*self.step + 1, self.step), self.rfecv.cv_results_["mean_test_score"])
        plt.show()