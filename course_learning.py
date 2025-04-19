import nltk
nltk.download()

paragrapgh="""
Zuckerberg briefly attended Harvard College, where he launched Facebook in February
 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. 
 Zuckerberg took the company public in May 2012 with majority shares. 
 He became the world's youngest self-made billionaire[a] in 2008, at age 23,
 and has consistently ranked among the world's wealthiest individuals. According to Forbes, 
 as of March 2025, Zuckerberg's estimated net worth stood at US$214.1 billion,
 making him the second richest individual in the world,[2] behind Elon Musk and before Jeff Bezos.

"""

sentence=nltk.sent_tokenize(paragrapgh)
words=nltk.word_tokenize(paragrapgh)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer=PorterStemmer()
for i in range(len(sentence)):
    words=nltk.word_tokenize(sentence[i])
    
    words=[stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    
    sentence[i]=" ".join(words)
    
    
print(sentence)
    
    
# for lammetizer

from nltk.stem import WordNetLemmatizer
sentence_2=nltk.sent_tokenize(paragrapgh)
lemmatizer=WordNetLemmatizer()

for i in range(len(sentence_2)):
    words= nltk.word_tokenize(sentence_2[i])
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words("english"))]
    sentence_2[i]=' '.join(words)
    
    
print(sentence_2)


# for BOW bag of words

# import 

from nltk import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
# convert the text to snetences
sentences_3=nltk.sent_tokenize(paragrapgh)
corpus=[]
for i in range(len(sentences_3)):
    #remove anything other than words
    text=re.sub('[^a-zA-Z]'," ",sentences_3[i])
    # to lower
    print(type(text))
    text=text.lower()
    #instead of word_tokenizer use split
    text=text.split()
    # clear stopwords
    text=[stemmer.stem(word) for word in text if word not in set(stopwords.words('english')) ]
    text=" ".join(text)
    corpus.append(text)
    
    # now use sklearn countvectorizer 
print(corpus)
cv=CountVectorizer()
x=cv.fit_transform(corpus).toarray()
  
    
print(x)
print(cv.vocabulary_)


from sklearn.feature_extraction.text import TfidfVectorizer
corpus_1=[]
sentences_4=nltk.sent_tokenize(paragrapgh)

for i in range(len(sentences_4)):
    text_1=re.sub('[^a-zA-Z]',' ',sentences_4[i])   # remove other than words
    text_1=text_1.lower()
    text_1=text_1.split() #instead of word_tokenize
    text_1=[lemmatizer.lemmatize(word) for word in text_1 if word not in set(stopwords.words("english"))]
    text_1=" ".join(text_1)
    corpus_1.append(text_1)
    
    
# instead of countvector we will use more accurate TfidfVectorizer


from sklearn.feature_extraction.text import TfidfVectorizer

tfVectorizer= TfidfVectorizer()

y=tfVectorizer.fit_transform(corpus_1).toarray()

    
    
    
    

   