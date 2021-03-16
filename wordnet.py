from nltk.corpus import wordnet

synonyms = []
antonyms = []

goodSyn= wordnet.synsets("good")
print(f'{goodSyn} \n')
for syn in goodSyn:
   
    for l in syn.lemmas():
      
        synonyms.append(l.name())
        if l.antonyms():
          
            antonyms.append(l.antonyms()[0].name())
