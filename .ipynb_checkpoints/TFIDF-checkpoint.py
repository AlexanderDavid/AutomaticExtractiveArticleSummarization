from typing import List, Callable, Dict       # Types to keep myself honest
from nltk import word_tokenize, sent_tokenize # Word and sentence tokenization from NLTK
from math import log                          # Logarithm function from math
from string import punctuation                # List of punctuation
import operator                               # Operator for fast tuple indexing
from collections import defaultdict           # DefaultDict so that there are no key_value errors

# Add some punctuation to the list of punctuation
punctuation = list(punctuation) + ["--"] + ["``"] + ["''"]

def idf(chapter: str, chapters: List[str]) -> Dict[str, int]:
    """Calculate and return all idf scores for each word in chapter 

    Args:
        chapter: The document to get the idf scores
        chapters: The corpus to use to "train" on.

    Returns:
        A dictionary containing the idf scores for all words in chapter

    """
    # Turn the chapter into a list of words
    words = [x for y in [word_tokenize(x) for x in sent_tokenize(chapter)] for x in y]
    # Strip all punctuation
    words_no_punc = list(filter(lambda x : x not in punctuation, words))
    # Create a new default dict with type int
    idf_dict = defaultdict(int)
    
    # For every word
    for word in words_no_punc:
        # Try to add the words idf score into the dictionary
        try:
            idf_dict[word] = log(len(chapters) / sum([1 if chapter.count(word) > 0 else 0 for chapter in chapters]))
        except ZeroDivisionError:
            pass
        
    # Return the dictionary
    return idf_dict

def tf(chapter: str) -> Dict[str, int]:
    """Calculate and return all of the tf scores for the chapter

    Args:
        chapter: The document to calculate the tf scores for

    Returns:
        A dictionary containing the idf scores for all words in chapter

    """
    # Create a list of all the words in the chapter
    words = [x for y in [word_tokenize(x) for x in sent_tokenize(chapter)] for x in y]
    # Strip out all of the the punctuation from the word list
    words_no_punc = list(filter(lambda x : x not in punctuation, words))
    # Return a dictionary of the frequencies for all words
    return {word: chapter.count(word) for word in words_no_punc}

def summarize(chapter: str, chapters: List[str], number_of_sents:int=5) -> List[str]:
    """Summarize a given chapter of a document

    Args:
        chapter: The document to summarize

    Returns:
        A dictionary containing the idf scores for all words in chapter

    """
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