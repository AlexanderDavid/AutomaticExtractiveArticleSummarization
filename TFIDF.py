from typing import List, Callable, Dict                           # Types to keep myself honest
from nltk import word_tokenize, sent_tokenize # Word and sentence tokenization from NLTK
from math import log
from glob import glob
from string import punctuation
from nltk import corpus
import operator
from collections import defaultdict

punctuation = list(punctuation) + ["--"] + ["``"] + ["''"]

def idf(chapter: str, chapters: List[str]):
    words = [x for y in [word_tokenize(x) for x in sent_tokenize(chapter)] for x in y]
    words_no_punc = list(filter(lambda x : x not in punctuation, words))
    idf_dict = defaultdict(int)
    for word in words_no_punc:
        try:
            idf_dict[word] = log(len(chapters) / sum([1 if chapter.count(word) > 0 else 0 for chapter in chapters]))
        except ZeroDivisionError:
            pass
        
    return idf_dict

def tf(chapter: str) -> Dict[str, int]:
    words = [x for y in [word_tokenize(x) for x in sent_tokenize(chapter)] for x in y]
    words_no_punc = list(filter(lambda x : x not in punctuation, words))
    return {word: chapter.count(word) for word in words_no_punc}

def summarize(chapter: str, chapters: List[str], number_of_sents:int=5) -> List[str]:
    tf_dict = tf(chapter)
    idf_dict = idf(chapter, chapters)
    sentences = sent_tokenize(chapter)
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        total_score = 0
        n = 0
        words = word_tokenize(sentence)
        for word in words:
            if word in punctuation:
                continue
                
            tf_score = tf_dict[word]
            idf_score = idf_dict[word]
            total_score += tf_score * idf_score
            n += 1
            
        sentence_scores.append((sentence, total_score / n, i))
        
    top_sents = list(reversed(sorted(sentence_scores, key=operator.itemgetter(1))))[:number_of_sents]
    return [x[0] for x in list(sorted(top_sents, key=operator.itemgetter(2)))]
