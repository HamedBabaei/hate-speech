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
    args = parser.parse_args()
    if args.input is None and args.output is None:
        parser.print_usage()
        exit()
    return args

def detector():
    args = get_args()
    input_dir = os.path.normpath(args.input)
    out_dir = os.path.normpath(args.output)
    en_hater = utils.load_pkl("en_hater.sav")
    es_hater = utils.load_pkl("es_hater.sav")
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
