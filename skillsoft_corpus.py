# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 22:33:00 2025

@author: Aboarab
"""
import nltk
from nltk.corpus import gutenberg

fileids=gutenberg.fileids()

for fileid in fileids:
    
    raw_data= gutenberg.raw(fileid)
    sentences=gutenberg.sents(fileid)
    words=gutenberg.words(fileid)
    
    num_sent=len(sentences)
    num_word=len(words)
    
# to get the freq of word in such file 
# use FreqDis

from nltk import FreqDist

words_1=gutenberg.words("whitman-leaves.txt")

freq=FreqDist(word.lower() for word in words_1)


common=freq.most_common(10)   # gives list((word,freq))
least=freq.hapaxes()[:10]  # gives words only

for word,freqnt in common:
    print(f'word:{word},freq:{freqnt}')

for word in least:
    frequency=freq[word]
    print(f'word:{word},freq:{frequency}')

#-------------------------------------------------------------
## brown

from nltk.corpus import brown

cats=brown.categories()

fileids_adv_categ=brown.fileids("adventure")

words_brown=brown.words("cn01")


# part of speech instead of nltk.pos_tag  tagged_words,sent,paras

sent_tag=brown.tagged_sents(fileids="cn01")
word_tag=brown.tagged_words(fileids="cn01")
para_tag=brown.tagged_paras(fileids="cn01")

print(sent_tag[:10])

print('--------------------')

print(word_tag[:10])
print('--------------------')

print(para_tag[:10])