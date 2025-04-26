import spacy
import nltk
from nltk.corpus import brown
nlp=spacy.load("en_core_web_sm")

print(nlp.pipe_names)

# we wille use the brown corpus for our project
fileid=brown.fileids(categories='adventure')[0]
# get raw text
raw_text=brown.raw(fileid)

# let us take the first file

doc=nlp(raw_text)

words=[token.text for token in doc]
positions=[token.pos_ for token in doc]
sents=[token.sent for token in doc]

# stopword

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.es.stop_words import STOP_WORDS as stop_words_es
words=[token.text for token in doc if token.is_stop==False]


#
