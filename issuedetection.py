from nrclex import NRCLex
from empath import Empath
import pandas as pd
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from heapq import nlargest

class IssueDetection:
    def __init__(self):
        self.lexicon = Empath()
        self.Attributes = ['help','money','sleep','aggression','envy','family','health','nervousness','suffering','optimism','religion','irritability','body','confusion','violence','neglect','strength','shame','affection','confusion','joy','dominant_personality','emotional','timidity','disappointment','work','rage','social_media','contentment']
        self.disorders=['PTSD','Anxiety','ED','BPD','Depression','SelfHarm']
        self.probs={}
        self.ranking = []
    def RandomForrestClassification(self,features):
        data = pd.read_csv('mentalhealth.csv')
        X = data.drop(['disorder'],axis = 1)
        where_are_NaNs = isnan(X)
        X[where_are_NaNs] = 0
        y = data['disorder']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = RandomForestClassifier(n_estimators=50,verbose=3,n_jobs=-1,random_state=50)
        classi=clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        pred = classi.predict_proba([features])
        pred1 = clf.predict([features])
        pred = pred.tolist()
        probs = {}
        probs['PTSD'] = pred[0][0]
        probs['Anxiety'] = pred[0][1]
        probs['ED'] = pred[0][2]
        probs['BPD'] = pred[0][3]
        probs['Depression'] = pred[0][4]
        probs['SelfHarm'] = pred[0][5]
        ThreeHighest = nlargest(3, probs, key = probs.get)
        temp = {}
        for i in ThreeHighest:
            temp[i]=probs[i]
            self.ranking.append(temp)
            temp = {}
        return self.ranking
        
    def analysis(self,text):
        emp_result = self.lexicon.analyze(text,normalize=True)
        lex_result = NRCLex(text)
        lex_result = lex_result.affect_frequencies
        lex_result.pop('anticip',None)
        for i in self.Attributes:
            lex_result[i] = emp_result[i]
        features=dict(sorted(lex_result.items(), key=lambda x:x[0].lower()))
        return self.RandomForrestClassification(list(features.values()))

