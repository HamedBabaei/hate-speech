from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from hatter import Hatter

def get_split_data(X, Y, indexes):
    X_split, y_split = [], []
    for index in indexes:
        X_split.append(X[index])
        y_split.append(Y[index])
    return X_split, y_split

def cross_validator(X, Y, cv, ngram_range, lang):
    skf = StratifiedKFold(n_splits=cv)
    cv_scores = []
    fold = 0
    for train_index, test_index in skf.split(X,Y):
        fold += 1
        X_train, y_train = get_split_data(X, Y, train_index)
        X_test, y_test = get_split_data(X, Y, test_index)
        model = Hatter(ngram_range=ngram_range, lang=lang)
        model.fit(X_train, y_train)
        preds = [str(model.predict_single(x)) for x in X_test]
        acc = accuracy_score(y_test, preds)
        print("FOLD-{}: Accuracy:{}".format(fold, acc))
        print("----------------------------------------------")
        cv_scores.append(acc)
        
    print("{}-Fold Crossvalidation-ensemble: {} +/- {},  FOLDS:{}".format(cv,
                                                                 np.mean(np.array(cv_scores)),
                                                                 np.std(np.array(cv_scores)),
                                                                 cv_scores))