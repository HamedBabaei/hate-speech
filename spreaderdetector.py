import codecs
import argparse
import os
import numpy as np
import utils
from tqdm import tqdm 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import  XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to dataset directory")
    parser.add_argument('-o', '--output', help="path to output directory")
    parser.add_argument('-t', '--train', help="path to train directory")
    args = parser.parse_args()
    if args.input is None and args.output is None:
        parser.print_usage()
        exit()
    return args

def get_ldsa(tfidf_matrix, Y, names, C=['1', '0']):
    ldsa = {}
    for index_t, t in tqdm(enumerate(names)):
        ldsa[t] = {'index_t':index_t}
        Wdt = tfidf_matrix[:, index_t]
        wdtc = {c:0 for c in C}
        for index_d, y in enumerate(Y):
            W = tfidf_matrix[index_d, index_t]
            wdtc[y] += W
        ldsa[t]['Wdt'] = sum(Wdt)
        for c in C:
            ldsa[t]['Wdt'+str(c)] = wdtc[c]/ldsa[t]['Wdt']
    return ldsa

def get_features(vectorizer, X, terms, ldsa):
    vectors = vectorizer.transform(X).toarray()
    features = {index:[[], [], []] for index, _ in enumerate(X)}
    for index_d, vector in tqdm(enumerate(vectors)):
        terms_d = [terms[index_t] for index_t, W in enumerate(vector) if W != 0]
        wd0, wd1 = [], []
        for term in terms_d:
            wd0.append(ldsa[term]['Wdt0'])
            wd1.append(ldsa[term]['Wdt1'])
        features[index_d][0] = wd0 
        features[index_d][1] = wd1
        features[index_d][2] = calculte_features(wd0) + calculte_features(wd1)
    return features

def calculte_features(lsts):
    #[avg, std, min, max, prob]
    X = np.array(lsts)
    features = [np.max(X), np.min(X), np.std(X), np.sum(X)/X.shape[0]]
    for q in list(range(0, 100, 1)):
        q = q/100
        features.append(np.quantile(X, q))
    return features

def train_en_hatespeech_detector(X, Y):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,3), 
                                       analyzer='char',
                                       sublinear_tf=True,
                                       smooth_idf=False)
    tfidf_vectorizer.fit(X)
    tfidf_matrix_train = tfidf_vectorizer.transform(X).toarray()
    feature_names = tfidf_vectorizer.get_feature_names()
    ldsa = get_ldsa(tfidf_matrix_train, Y, feature_names, C=['1','0'])
    features = get_features(tfidf_vectorizer, X, feature_names, ldsa)
    X_train_ldsa = [features[2] for _,features in features.items()]

    estimator = ExtraTreesClassifier(n_estimators=100)
    estimator.fit(X_train_ldsa, Y)

    sk = SelectFromModel(estimator, prefit=True)
    X_train_ldsa_sk = sk.transform(X_train_ldsa)
    
    en_model = SVC(C=0.1, kernel='linear')
    en_model.fit(X_train_ldsa_sk, Y)
    return en_model, sk, ldsa, tfidf_vectorizer, feature_names

def en_predictor(X, en_model, sk, ldsa, tfidf_vectorizer, feature_names):
    features = get_features(tfidf_vectorizer, X, feature_names, ldsa)
    X_train_ldsa = [features[2] for _,features in features.items()]
    X_train_ldsa_sk = sk.transform(X_train_ldsa)
    return en_model.predict(X_train_ldsa_sk)[0]

def train_es_hatespeech_detector(X, Y):
    clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(2,4), 
                                          analyzer='char')),
                ('clf', XGBClassifier())])
    clf.fit(X, Y)
    return clf

def detector():
    args = get_args()
    input_dir = os.path.normpath(args.input)
    out_dir = os.path.normpath(args.output)
    train_dir = os.path.normpath(args.train)

    en = utils.get_data(os.path.join(train_dir, 'en'))
    es = utils.get_data(os.path.join(train_dir, 'es'))
    x_en, y_en = en['tweet'].tolist(), en['label'].tolist()
    x_es, y_es = es['tweet'].tolist(), es['label'].tolist()

    x_en = [X.replace("<TWEET>", ' \n ') for X in tqdm(x_en)]
    x_es = [X.replace("<TWEET>", ' \n ') for X in tqdm(x_es)]

    es_model = train_es_hatespeech_detector(x_es, y_es)
    en_model, en_sk, en_ldsa, en_tfidf_vectorizer, en_feature_names = train_en_hatespeech_detector(x_en, y_en)

    utils.mkdir(out_dir)
    for language_dir in os.listdir(input_dir):
        input_dir_path = os.path.join(input_dir , language_dir)
        out_dir_path = os.path.join(out_dir , language_dir)
        utils.mkdir(out_dir_path)
        for user in os.listdir(input_dir_path):
            print(language_dir , "::::Working on user: ", user )
            user_tweets = '\n '.join(utils.read_xml(os.path.join(input_dir_path , user)))
            if language_dir == 'en':
                pred = en_predictor([user_tweets], en_model, en_sk, en_ldsa, 
                                     en_tfidf_vectorizer, en_feature_names)
            else:
                pred = es_model.predict([user_tweets])
            utils.save_xml(os.path.join(out_dir_path, user) , user , str(language_dir) , str(pred[0]) )
            print("Results saved to " , str(os.path.join(out_dir_path, user)))
            print("-------------------------------------------------")

detector()