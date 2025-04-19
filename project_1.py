# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:58:07 2025

@author: Aboarab
"""
import pandas as pd
import numpy as np


data=pd.read_csv("smsspamcollection",sep="\t",names=["label","message"])


import nltk
nltk.download()
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from nltk import re
from nltk.corpus import stopwords
lemmatizer= WordNetLemmatizer()
vect=CountVectorizer()


X=data["message"]
y=pd.factorize(data["label"])[0]

corpus=[]

# create the BOW
for i in range(len(X)):
 
    text=re.sub("[^a-zA-Z]"," ",X[i])
    text=text.lower()
    text=text.split()
    text=[lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text=' '.join(text)
    corpus.append(text)
    
    
    
    
message=vect.fit_transform(corpus).toarray()


#now for the classification model training

x_train,x_test,y_train,y_test=train_test_split(message,y,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB

classifier= MultinomialNB()

classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

print(classifier.score(x_train,y_train))
print(classifier.score(x_test,y_test))


email=['''
Dear Beneficiary,

I am certain this message might come to you as a surprise. Nevertheless,
I humbly ask you to give me your attention and hear me well. I am Hon
Mr.Anthony Norman Albanese. The current Prime Minister of Australia. I
want to inform you that the matter of your fund was brought to my desk
this today, because Michele Bullock The Governor of, Reserve Bank of
Australia .said that they will divert your fund worth a sum of
AUD$12,000,000,000.00 Twelve Billion United Australian Dollars to
Government
Treasuries Account just because you refused to pay the sum of $250 AUD
Activation fee which will permit you a daily withdrawal limit of
US$10,000 aud three times daily from any ATM Machine around you.

Please take note that the AUD$12,000,000,000.00 deposited with the
Reserve Bank of Australia won by your e-mail address via online active
email address selected from the sponsor of Coca Cola Company award. It
was registered with your email address by the Lottery Board Executive
Directors Of Coca-Cola Company Australia Branch, As Prime Minister of
Australia, I told Mrs.Michele Bullock the Governor to reduce the
Activation fee to $50 usd to help you afford the fee. After the
The Diplomatic Agent delivers the ATM Visa Card then he will follow you to
any ATM MACHINE center around you so you can withdraw with your ATM Card
and then you balance him the remaining $200 aud.

The Reserve Bank of Australia has given you the grace of sending only
$50.00 of the charges/fee which will be only $50 for now, then once you
confirm your AUD$12,000,000,000.00, you can then pay the remaining
balance of $200.

I want to personally assure you once again that you will have every
opportunity to smile and be happy to organize a big party to celebrate with
your family.

I am hereby protecting your interest as the Prime Minister of Australia
to make sure all goes well because this is a huge amount of money, we
cannot wish you to lose. Please know that you have from now till the end
of two days to effect the payment for activation of your ATM Card
Account Number: 8020253465.

The payment approval documents have been acquired for the smooth
delivery of your ATM visa Card of $12,000,000,000.00. You are advised to
buy a $50 aud Steam Wallet Card or Apple iTunes Card to enable the
Reserve Bank of Australia activate the account for you to be able to
withdraw immediately you receive the ATM Card to your home address.

Your quick attention is needed as soon as you read this email and you
should avoid anything that will make you lose this fund.

Thanks, and God bless.
Best Regards,
Hon. Mr.Anthony Norman Albanese
Prime Minister of Australia


'''
] 

sentences= nltk.sent_tokenize(email[0])
email_1=[]
for i in range(len(sentences)):
    
    email=re.sub("[^a-zA-Z]"," ",sentences[i])
    email=email.lower()
    email=email.split()
    email=[lemmatizer.lemmatize(word) for word in email if word not in set(stopwords.words('english'))]
    email=' '.join(email)
    email_1.append(email)
    
x_email=vect.transform(email_1).toarray()
result=classifier.predict(x_email)
final=np.argmax(result)                                  










