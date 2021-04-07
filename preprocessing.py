import re
import string
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')
#from spellchecker import SpellChecker
from nltk.stem import PorterStemmer, WordNetLemmatizer

class Preprocessing:
    """
    this class is provided to do text preprocessing like cleaning and normalizing
    :params: permission to each method of preprocessign and if it's true, that method will be done and if false will be skiped
    """

    def __init__(self, lowercasing=True, number_removing=True, punctuation_removing=True, whitespaces_removing=True,
                 stopwords_removing=True, spell_checking=True, HTML_removing=True, URL_removing=True,
                 emoji_removing=True,stemming=True, lematizing = False, contraction_expanding=True):

        self.configuration = {
            0: {"permission": HTML_removing, "method": self.HTML_remover},
            1: {"permission": URL_removing, "method": self.URL_remover},
            2: {"permission": emoji_removing, "method": self.emoji_remover},
            3: {"permission": lowercasing, "method": self.lowercaser},
            4: {"permission": number_removing, "method": self.number_remover},
            5: {"permission": contraction_expanding, "method": self.contraction_expander},
            6: {"permission": punctuation_removing, "method": self.punctuation_remover},
            7: {"permission": whitespaces_removing, "method": self.whitespaces_remover},
            8: {"permission": spell_checking, "method": self.spell_checker},
            9: {"permission": stopwords_removing, "method": self.stopwords_remover},
            10: {"permission": stemming, "method": self.stemmer},
            11: {"permission": lematizing, "method": self.lematizer}
        }

    def clean(self, Text, list_output=False, lsts = None):
        """
        for each method, if it's permission is True, pass the text to it and the output of that method is the input of next method
        :param Text: a plain text
        :return: preprocessed plain text or list of token if list_output True (defualt is plain text)
        """
        for config_no in range(0, len(self.configuration)):
            Text = self.configuration[config_no]['method'](Text, self.configuration[config_no]['permission'])
        if list_output:
            return Text.split()
        elif lsts != None:
            for char in lsts:
                Text = Text.replace(char, "")
        return Text

    def tokenizer(self, X):
        """
        tokenizing text
        :param X: a plain text
        :return: a list of tokens
        """
        return word_tokenize(X)

    def HTML_remover(self, X, permission):
        """
        removing HTML tags
        :param X: a plain text
        :return: a plain text without HTML tags
        """
        if permission == True:
            cleaner = re.compile('<.*?>')
            cleantext = re.sub(cleaner, '', X)
            return cleantext
        else:
            return X

    def URL_remover(self, X, permission):
        """
        removing URLs
        :param X: a plain text
        :return: a plain text without URLs
        """
        if permission == True:
            cleaner = re.compile('http\S+')
            cleantext = re.sub(cleaner, '', X)
            return cleantext
        else:
            return X

    def emoji_remover(self, X, permission):
        """
        removing emojies
        :param X: a plain text
        :return: a plain text without emojies
        """
        if permission == True:
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', X)
        else:
            return X

    def lowercaser(self, X, permission):
        """
        lowercasing
        :param X: a plain text
        :return: a plain text in which all characters are lowercased
        """
        if permission == True:
            return X.lower()
        else:
            return X

    def number_remover(self, X, permission):
        """
        removing numberss
        :param X: a plain text
        :return: a plain text without numbers
        """
        if permission == True:
            return re.sub(r'\d+', '', X)
        else:
            return X

    def punctuation_remover(self, X, permission):
        """
        removing punctuations
        :param X: a plain text
        :return: a plain text without punctuations
        """
        if permission == True:
            return X.translate(str.maketrans('', '', string.punctuation))
        else:
            return X

    def whitespaces_remover(self, X, permission):
        """
        removing extra white spaces
        :param X: a plain text
        :return: a plain text without extra white spaces
        """
        if permission == True:
            return ' '.join(X.split())
        else:
            return X

    def stopwords_remover(self, X, permission):
        """
        removing stopwords
        :param X: a plain text
        :return: a plain text without stopwords
        """
        if permission == True:
            tokens = self.tokenizer(X)
            filtered_words = [word for word in tokens if word not in stopwords.words('english')]
            return ' '.join(filtered_words)
        else:
            return X

    def contraction_expander(self, X, permission):
        """
        expanding contractions like 'it's' to 'it is'
        :param X: a plain text
        :return: a plain text without contractions
        """
        #if permission == True:
        #    expanded = [contractions.fix(word) for word in X.split()]
        #    return ' '.join(expanded)
        #else:
        #    return X
        return X

    def spell_checker(self, X, permission):
        """
        correcting spells like 'exemple' to 'example'
        :param X: a plain text
        :return: a plain text without contractions
        """
        if permission == True:
            spell = SpellChecker()
            words = spell.split_words(X)
            spell_checked = [spell.correction(word) for word in words]
            return ' '.join(spell_checked)
        else:
            return X

    def stemmer(self, X, permission):
        """
        stemming like 'apples' to 'apple'
        :param X: a plain text
        :return: a stemmed plain text
        """
        if permission == True:
            tokens = self.tokenizer(X)
            ps = PorterStemmer()
            stemmed = [ps.stem(token) for token in tokens]
            return ' '.join(stemmed)
        else:
            return X

    def lematizer(self, X, permission):
        """
        lematization like 'apples' to 'apple'
        :param X: a plain text
        :return: a lematized plain text
        """
        if permission == True:
            return ' '.join([WordNetLemmatizer().lemmatize(token) 
                             for token in self.tokenizer(X)])
        else:
            return X
            
   
