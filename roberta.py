from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelWithLMHead
import utils
import numpy as np

class Roberta:

    def __init__(self, lang='en', summary_ratio=0.1):
        self.summary_ratio = summary_ratio
        if lang == 'en':
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        else:
            self.roberta_tokenizer = AutoTokenizer.from_pretrained("skimai/spanberta-base-cased")
            self.roberta_model = AutoModelWithLMHead.from_pretrained("skimai/spanberta-base-cased")
    
    def transform(self, X):
        return [self.transform_sample(X, ratio=self.summary_ratio) for X in X]

    def transform_sample(self, text, ratio):
        try:
            summary = utils.get_summary(text, ratio=ratio)
            summary = utils.cleaning(summary)
            inputs = self.roberta_tokenizer(summary, return_tensors="pt")
            outputs = self.roberta_model(**inputs)
            vec = np.mean(outputs[0].detach().numpy(),  axis=1)[0]
            return vec
        except:
            summary = utils.get_summary(text, ratio=0.05)
            summary = utils.utils.cleaning(summary)
            inputs = self.roberta_tokenizer(summary, return_tensors="pt")
            outputs = self.roberta_model(**inputs)
            vec = np.mean(outputs[0].detach().numpy(),  axis=1)[0]
            return vec
