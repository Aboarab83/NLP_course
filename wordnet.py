from nltk.corpus import wordnet

synsets=wordnet.synsets("car")

# for synset in synsets:
#     print(f'synset:{synset.name()}\t '
#           f'meaning:{synset.definition()}\t pos:{synset.pos()}\t '
#           f'examples:{synset.examples()}')


print(synsets[0].hypernyms())
print(synsets[0].hyponyms())
print(synsets[0].part_holonyms())
print(synsets[0].lemmas()[0].antonyms())
print(synsets[0].part_meronyms())