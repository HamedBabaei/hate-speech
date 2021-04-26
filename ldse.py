from tqdm import tqdm
import numpy as np

class LDSE:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
    
    def fit(self, X, Y, C=['1', '0']):
        self.vectorizer.fit(X)
        matrix = self.vectorizer.transform(X).toarray()
        self.names = self.vectorizer.get_feature_names()
        self.embedding = self.get_ldse(matrix, Y, C=C)
        
    def transform(self, X, C1=False):
        vectors = self.vectorizer.transform(X).toarray()
        features = {index:[[], [], []] for index, _ in enumerate(X)}
        for index_d, vector in enumerate(vectors):
            terms_d = [self.names[index_t] for index_t, W in enumerate(vector) if W != 0]
            features[index_d][0] = [self.embedding[term]['Wdt0'] for term in terms_d]
            features[index_d][1] = [self.embedding[term]['Wdt1'] for term in terms_d]
            if C1:
                features[index_d][2] = self.get_distribution(features[index_d][1])
            else:
                features[index_d][2] = self.get_distribution(features[index_d][0])
        return [F[2] for _,F in features.items()]
    
    def get_ldse(self, matrix, Y, C):
        ldse = {}
        for index_t, t in tqdm(enumerate(self.names)):
            ldse[t] = {'index_t':index_t}
            Wdt = matrix[:, index_t]
            wdtc = {c:0 for c in C}
            for index_d, y in enumerate(Y):
                W = matrix[index_d, index_t]
                wdtc[y] += W
            ldse[t]['Wdt'] = sum(Wdt)
            for c in C:
                ldse[t]['Wdt'+str(c)] = wdtc[c]/ldse[t]['Wdt']
        return ldse
    
    def get_distribution(self, lsts):
        X = np.array(lsts)
        features = [np.max(X), np.min(X), np.std(X), np.sum(X)/X.shape[0]]
        for q in list(range(0, 100, 1)):
            q = q/100
            features.append(np.quantile(X, q))
        return features
    