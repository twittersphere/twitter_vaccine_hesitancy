from nltk.tokenize import RegexpTokenizer

class NLPTools:
    def __init__(self) -> None:
        self.TOKENIZER = RegexpTokenizer(r'\w+')