import argparse
import os
import utils
from hater import Hater

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', help="path to train directory")
    args = parser.parse_args()
    if args.train is None:
        parser.print_usage()
        exit()
    return args

args = get_args()
train_dir = os.path.normpath(args.train)

en = utils.get_data(os.path.join(train_dir, 'en'))
es = utils.get_data(os.path.join(train_dir, 'es'))
x_en, y_en = en['tweet'].tolist(), en['label'].tolist()
x_es, y_es = es['tweet'].tolist(), es['label'].tolist()

x_en = [X.replace("<TWEET>", ' \n ') for X in tqdm(x_en)]
x_es = [X.replace("<TWEET>", ' \n ') for X in tqdm(x_es)]

en_hater = Hater(ngram_range=(2, 3), lang='en')
en_hater.fit(x_en, y_en)

es_hater = Hater(ngram_range=(3, 4), lang='es')
es_hater.fit(x_es, y_es)

utils.save_pkl("eg_hater.sav", en_hater)

utils.save_pkl("es_hater.sav", es_hater)
