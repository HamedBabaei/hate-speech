import codecs
import argparse
import os
import numpy as np
import utils
from tqdm import tqdm 
from hater import Hater

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

    en_hater = Hater(ngram_range=(2, 3), lang='en')
    en_hater.fit(x_en, y_en)

    es_hater = Hater(ngram_range=(3, 4), lang='es')
    es_hater.fit(x_es, y_es)

    utils.mkdir(out_dir)
    for language_dir in os.listdir(input_dir):
        input_dir_path = os.path.join(input_dir , language_dir)
        out_dir_path = os.path.join(out_dir , language_dir)
        utils.mkdir(out_dir_path)
        for user in os.listdir(input_dir_path):
            print(language_dir , "::::Working on user: ", user )
            user_tweets = '\n '.join(utils.read_xml(os.path.join(input_dir_path , user)))
            if language_dir == 'en':
                pred = en_hater.predict_single(user_tweets)
            else:
                pred = es_hater.predict_single(user_tweets)
            utils.save_xml(os.path.join(out_dir_path, user) , user , str(language_dir) , str(pred))
            print("Results saved to " , str(os.path.join(out_dir_path, user)))
            print("-------------------------------------------------")

detector()
