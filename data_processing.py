import json
import string

"""import ibm watson api"""
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions

"""import nltk"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

"""import matlabplot visualization"""


paper = open('maas-ismm12-gpugc.txt','r')
paper_txt = paper.read()

"""data pre-processing"""

"""remove punctuation"""
paper_txt = paper_txt.translate(None, string.punctuation)

"""get rid of non-ascii-able character"""
printable = set(string.printable)
paper_txt = filter(lambda x: x in printable, paper_txt)

# lemmatizing word for same time pattern
lemmatizer = WordNetLemmatizer()
paper_txt = lemmatizer.lemmatize(paper_txt)

tokens = word_tokenize(paper_txt)

# remove stop-word token
stopwords.words('english')

for token in tokens:
    token = token.lower()

clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english') or token == "We" or token == "The":
        clean_tokens.remove(token)
        

fdist = FreqDist(word.lower() for word in clean_tokens)
"""
for key, val in fdist.items():
    print(str(key) + ':' + str(val))
"""

fdist.plot(20, cumulative=False)

"""
nlu = NaturalLanguageUnderstandingV1(
    username='2372a4be-eb9b-4e40-9736-a49952a84a43',
    password='YxsT7SqUisuu',
    version='2018-03-16'
)

response = nlu.analyze(
    text = paper_txt,
    features = Features(
        entities = EntitiesOptions(
            emotion = True,
            sentiment = True,
            limit = 2
        ),
        keywords = KeywordsOptions(
            emotion = True,
            sentiment = True,
            limit = 2
        )
    )
)

print(json.dumps(response, indent=2))
"""