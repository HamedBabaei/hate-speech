from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from roberta import Roberta
from ldse import LDSE

class Hatter:
    def __init__(self, ngram_range, lang='en', summary_ratio=0.1):
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, 
                                     analyzer='char',
                                     sublinear_tf=True,
                                     smooth_idf=False)
        self.ldse_model = LDSE(vectorizer=vectorizer)
        self.label0_model = SVC(C=0.1, kernel='linear')
        self.label1_model = SVC(C=0.1, kernel='linear')

        self.bert_transformer = Roberta(lang=lang, summary_ratio=summary_ratio)
        self.bert_model   = SVC(C=0.1, kernel='linear')        

    def fit(self, X, y, C=['1', '0']):
        self.ldse_model.fit(X, y, C=C)
        label0_representation = self.ldse_model.transform(X, C1=False)
        label1_representation = self.ldse_model.transform(X, C1=True)
        bert_representation   = self.bert_transformer.transform(X)

        self.label0_model.fit(label0_representation, y)
        self.label1_model.fit(label1_representation, y)
        self.bert_model.fit(bert_representation, y)

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def predict_single(self, X):
        predicts = {'0':0, '1':0}
        
        F0 = self.ldse_model.transform([X], C1=False)
        F1 = self.ldse_model.transform([X], C1=True)
        F2 = self.bert_model.transform([X])

        predicts[self.label0_model.predict(F0)[0]] += 1
        predicts[self.label1_model.predict(F1)[0]] += 1
        predicts[self.bert_model.predict(F2)[0]] += 1
        
        return 0 if predicts['0'] > predicts['1'] else 1
