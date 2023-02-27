from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

class Classifier:
    
    #
    #
    # Logistic Regression CV
    def logistic_init(self):
        self.log_reg = make_pipeline(StandardScaler(), LogisticRegressionCV(class_weight='balanced', scoring='balanced_accuracy'))
        
    def logistic_fit(self, x, y):
        self.log_reg.fit(x, y)
        
    def logistic_pred(self, x):
        return self.log_reg.predict(x)
    
    def logistic_score(self, x, y):
        return self.log_reg.score(x, y)
    
    def get_logistic(self):
        return self.log_reg
    
    #
    #
    #Support Vector Classification (SVC)
    def svc_init(self):
        self.svc = make_pipeline(StandardScaler(), SVC())
    
    def svc_fit(self, x, y):
        self.svc.fit(x, y)
        
    def svc_pred(self, x):
        return self.svc.predict(x)
    
    def svc_score(self, x, y):
        return self.svc.score(x, y)
    
    def get_svc(self):
        return self.svc