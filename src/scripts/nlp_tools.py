from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag_sents
from tqdm import tqdm


class NLPTools:
    def __init__(self) -> None:
        self.TOKENIZER = RegexpTokenizer(r'\w+')

        self.pos_encoding = {'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',
                             'PRP': 'n', 'PRP$': 'n', 'VB': 'v', 'VBD': 'v',
                             'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
                             'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 'RB': 'r',
                             'RBR': 'r', 'RBS': 'r'}

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def sub_tokenize_and_remove_stopwords(self, text):
        sub_cleaned_text = []
        for word in self.TOKENIZER.tokenize(text.lower()):
            if len(word) > 1 and word not in self.stop_words:
                sub_cleaned_text.append(word)

        return ' '.join(sub_cleaned_text)

    def tokenize_and_remove_stopwords(self, text_list):
        filtered_text = []
        for text in tqdm(text_list):
            sub_cleaned_text = self.sub_tokenize_and_remove_stopwords(text)
            filtered_text.append(sub_cleaned_text)

        return filtered_text

    def apply_pos_tagging(self, text_list):
        return pos_tag_sents([text.split() for text in text_list])

    def lemmatize_text(self, text_list):
        filtered_text = []
        for text in tqdm(text_list):
            sub_cleaned_text = []
            for word in text:
                if self.pos_encoding.get(word[1], False):
                    lemmatized_word = self.lemmatizer.lemmatize(
                        word[0], pos=self.pos_encoding[word[1]])
                else:
                    lemmatized_word = self.lemmatizer.lemmatize(word[0])

                sub_cleaned_text.append(lemmatized_word)
            filtered_text.append(' '.join(sub_cleaned_text))

        return filtered_text
