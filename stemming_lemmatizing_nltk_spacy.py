import nltk
import spacy
from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer, RegexpStemmer, word_tokenize
from nltk.corpus import wordnet
port=PorterStemmer()
snow=SnowballStemmer("english")
lanc=LancasterStemmer()
word="exploring"
print(f"word:{word}\t port:{port.stem(word)}\t"
      f"snow:{snow.stem(word)}\tlancaster:{lanc.stem(word)}")

from nltk import WordNetLemmatizer,re
text_1="I like exploring the forest and being happy , see mice and mountains walking rapidly"

lem=WordNetLemmatizer()
para=[]

text=text_1.lower()
text=text.split()
text_lem=[lem.lemmatize(word) for word in text if word not in set(nltk.corpus.stopwords.words("english"))]
text_lem=" ".join(text)
print(text_lem)

# it is better ot lemmatize it after pos_tag
from nltk import pos_tag

# to use pos_tag we have to word_tokenize
words=nltk.word_tokenize(text_1)
pos_tags=pos_tag(words,tagset="universal")

# since lemmatizer work with the wordnet.xx format we will create custom def for that because the nltk does not contain explain() like spacy

def get_tag_explain(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


lemmatization=[lem.lemmatize(word,get_tag_explain(tag))for  (word,tag) in pos_tags ]
print(lemmatization)

#spacy only lemma and pos

import spacy
nlp=spacy.load("en_core_web_sm")
doc=nlp(text_1)

lemma=[token.lemma_ for token in doc]
tagss=[token.pos_ for token in doc]
print(lemma)
print(tagss)


from spellchecker import SpellChecker

sp_ch=SpellChecker()

text_wrong=" I willl go too the zoo for see the zepra"
words=text_wrong.split()
corrected_words=[sp_ch.correction(word) for word in words]
new_text=" ".join(corrected_words)



# display tag in spacy token.pos_

# named entity in spacy using displacy

from spacy import displacy

text_ent="Mohamed was born at Kafr eldawar, buhyra at 1983 "

doc_ent=nlp(text_ent)

from spacy import displacy
# print(displacy.render(doc_ent,style="ent"))
# print(displacy.render(doc_ent,style="dep"))

# cutom ERN
from spacy.tokens import Span
# doc_ent.spans["custom_spans"]=[
#     Span(doc_ent,4,6,"LOC"),
#     Span(doc_ent,7,8,"GPE")
# ]
# print(displacy.render(doc_ent,style="span",options={"spans_key":"custom_spans"}))
# to combine default ent and custom span we will use span style

# extract the default spans
default_span=doc_ent.ents

custom_spans=[Span(doc_ent,4,6,"LOC"),
    Span(doc_ent,7,8,"GPE")]
# combine

doc_ent.spans['all_spans']=list(default_span)+custom_spans
# render

print(doc_ent.spans)
print(displacy.render(doc_ent,style="span",options={'spans_key':'all_spans'}))


