import nltk
# from summarizer import Summarizer
import pprint
import itertools
import re
import pke
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import requests
import json
import re
import random
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
from nltk.corpus import wordnet as wn
import random
from youtube_transcript_api import YouTubeTranscriptApi
from deepmultilingualpunctuation import PunctuationModel

nltk.download('stopwords')
nltk.download('popular')

def get_keywords(text):
    keywords = []
    client = pke.unsupervised.MultipartiteRank()
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    client.load_document(input = text, stoplist=stoplist)
    pos = {'PROPN'}
    #pos = {'VERB', 'ADJ', 'NOUN'}
    client.candidate_selection(pos=pos)
    client.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
    words = client.get_n_best(n=20)
    for word in words:
        keywords.append(word[0])
    return keywords

def sorting(e):
    return len(e)

def get_sentences(text):
    keywords = get_keywords(text)
    tokenized_sentences = sent_tokenize(text)
    client = KeywordProcessor()
    sentences = {}
    for word in keywords:
        client.add_keyword(word)
        sentences[word] = []
    for sentence in tokenized_sentences:
        keywords = client.extract_keywords(sentence)
        for key in keywords:
            sentences[key].append(sentence)
        for key in sentences.keys():
            sentences[key].sort(key=sorting)
    return sentences

def get_wordsense(sentence, word):
    word= word.lower()
    
    if len(word.split())>0:
        word = word.replace(" ","_")
    
    synsets = wn.synsets(word,'n')
    if synsets:
        wup = max_similarity(sentence, word, 'wup', pos='n')
        adapted_lesk_output =  adapted_lesk(sentence, word, pos='n')
        lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

def get_wordnet(ws, word):
    words = []
    orig = word
    word = word.lower()
    hyp = ws.hypernyms()
    if len(hyp) == 0:
        return words
    for item in hyp[0].hyponyms():
        name = item.lemmas()[0].name()
        if(name == word or name == orig):
            continue
        name = name.replace(" ", "_")
        if name is not None and name not in words:
            words.append(name)
    return words

def get_conceptnet(word):
    word = word.lower()
    original_word= word
    if (len(word.split())>0):
        word = word.replace(" ","_")
    distractor_list = [] 
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
    obj = requests.get(url).json()

    for edge in obj['edges']:
        link = edge['end']['term'] 

        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)
                   
    return distractor_list

def get_similar(text):
    sentences = get_sentences(text)
    q_a = []
    for key in sentences.keys():
        similar = []
        wordsense = get_wordsense(sentences[key][0], key)
        if wordsense:
            similar = get_wordnet(wordsense, key)
        if len(similar) < 3:
            similar = similar + get_conceptnet(key)
        sentence = sentences[key][0].replace(key, "_____").replace(key.capitalize(), "_____")
        q_a.append({
            "question": sentence,
            "answer": key,
            "other_choices": similar,
        })
    return q_a
            
def generate_questions(text):
    letters = ["A", "B", "C", "D"]
    q_a = get_similar(text)
    for question in q_a:
        choices = [question['answer'].capitalize()]+ question['other_choices']
        if(len(choices) > 3):
            answers = choices[:4]
            random.shuffle(answers)
            print(f"Fill in the blank: {question['question']}")
            for i in range(4):
                try:
                    print(f"{letters[i]}: {answers[i]}")
                except:
                    letters = letters
def get_youtube_transcript(source):
    transcript = YouTubeTranscriptApi.get_transcript(source)
    script = ""
    for i in transcript:
        script += f"{i['text']} "
    model = PunctuationModel()
    result = model.restore_punctuation(script)
    return result

sample_text = "The Revolutionary War was a war unlike any other — one of ideas and ideals, that shaped “the course of human events.” With 165 principal engagements from 1775-1783, the Revolutionary War was the catalyst for American independence. Our inalienable rights, as laid out in the Declaration of Independence, were secured by George Washington and his army at the Siege of Boston, the American victory at Princeton and the stunning British surrender at Yorktown. Explore the battlefields and personalities from this pivotal time in American history."

# print(generate_questions(sample_text))
print(get_youtube_transcript('KGhacRRMnDw'))