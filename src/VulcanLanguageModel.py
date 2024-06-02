from decimal import Decimal, getcontext, DivisionByZero, InvalidOperation

import os
import nltk
import string
nltk.download('punkt')

getcontext().prec = 50


def get_path(dir: str) -> str:
    # Get the directory of the current file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the absolute path of the project directory
    project_directory = os.path.abspath(os.path.join(current_file_directory, os.pardir))
    # Get the absolute path to the data folder
    return os.path.join(project_directory, dir) + os.sep


TEXTS_PATH = get_path('data')
formatted_bigram_prob = None
formatted_trigram_prob = None


def init(*args):
    global formatted_bigram_prob
    global formatted_trigram_prob
    
    if formatted_bigram_prob is not None:
        print("Vulcan Language Model is already initialized...")
        return

    print("Initializing Vulcan Language Model...")
    text_docs, test_sentences = __load_docs()

    print("Words in training set: " + str(len(text_docs)))
    print("Counting frequencies...")

    unigram_word_dict, bigram_word_dict, unique_words, trigram_word_dict = __count_freq(text_docs)




    print("Counting probabilities...")
    bigram_word_prob = __count_bigram_prob(unigram_word_dict, bigram_word_dict, unique_words)
    trigram_word_prob = __count_trigram_prob(unigram_word_dict, unique_words, trigram_word_dict)
    # print(trigram_word_prob)

    formatted_bigram_prob = __transform(bigram_word_prob)
    formatted_trigram_prob = __trigram_transform(trigram_word_prob)

    print("Init complete")
    
    if (args != ()):
        bigram_perplexity = __calculate_bigram_perplexity(test_sentences, bigram_word_prob, unigram_word_dict)
        print("Bigram perplexity:", bigram_perplexity)
        
        trigram_perplexity = __calculate_trigram_perplexity(test_sentences, trigram_word_prob, bigram_word_prob, unigram_word_dict)
        print("Trigram perplexity:", trigram_perplexity)
        
        bigram_accuracy = __compute_bigram_accuracy(bigram_word_dict, test_sentences)
        print("Bigram Accuracy: " + str(bigram_accuracy))
        
        trigram_accuracy = __compute_trigram_accuracy(trigram_word_dict, test_sentences)
        print("Trigram Accuracy: " + str(trigram_accuracy))
        
        




def predict(word: str) -> tuple[str, str, str]:
    if formatted_bigram_prob is None:
        raise Exception("predict() was called before init().")

    def get_word(arr: list, index: int):
        try:
            return arr[index][0]
        except Exception:
            return ''

    try:
        r: list[tuple[str, float]] = sorted(formatted_bigram_prob[word].items(), key=lambda x: x[1], reverse=True)
    except KeyError:
        return ('', '', '')

    results = []

    for tup in r:
        if ((tup[0] in string.punctuation) is False) and ((tup[0] in blacklist) is False):
            results.append(tup)

    return (get_word(results, 0), get_word(results, 1), get_word(results, 2))


def trigram_predict(word1, word2):
    if formatted_bigram_prob is None:
        raise Exception("predict() was called before init().")
    
    def get_word(arr: list, index: int):
        try:
            return arr[index][0]
        except Exception:
            return ''
        
    try:
        r = sorted(formatted_trigram_prob[(word1, word2)].items(), key=lambda x: x[1], reverse=True)
    except KeyError:
        return ('', '', '')
    
    results = []

    for tup in r:
        if ((tup[0] in string.punctuation) is False) and ((tup[0] in blacklist) is False):
            results.append(tup)

    return (get_word(results, 0), get_word(results, 1), get_word(results, 2))



blacklist = '“”‘’ …_—﻿™•£·'
punctuation = "!\"#$%&()*+, ./:;<=>?@[]^_\\`{|}~"


def __parse_sentence(sentence: str) -> list[str]:
    for char in blacklist:
        sentence = sentence.replace(char, "")

    output = []
    for word in sentence.split(" "):
        if not word:
            continue
        

        if word[-1] in punctuation:
            output.append(word[:-1])
            output.append(word[-1])
        else:

            output.append(word)

    return output


def __load_docs() -> tuple[list[str], list[str]]:
    training_set_words: list[str] = []
    testing_set_sentences: list[str] = []

    docs: list[str] = []
    for file in os.listdir(TEXTS_PATH):
        with open(TEXTS_PATH + file, "r", encoding="utf-8") as f:
            docs.append(nltk.sent_tokenize(f.read()))

    all_words: list[str] = []

    for book in docs:
        training_set_words.extend(book[: int(len(book) * 0.8)])
        testing_set_sentences.extend(book[int(len(book) * 0.8):])

    for sentence in training_set_words:
        words = __parse_sentence(sentence)
        for i in range(len(words)):
            words[i] = words[i].lower()
        all_words.extend(words)

    return all_words, testing_set_sentences


def __count_freq(word_list: list[str]) -> tuple[dict[str, int], dict[str, int], set[str]]:
    unique_words = set()
    unigram_word_dict = {}

    for word in word_list:
        unique_words.add(word)
        try:
            unigram_word_dict[word] += 1
        except KeyError:
            unigram_word_dict[word] = 1

    # Count frequency of pair words
    bigram_word_dict = {}
    for i in range(len(word_list) - 1):
        word = word_list[i] + " " + word_list[i + 1]
        try:
            bigram_word_dict[word] += 1
        except KeyError:
            bigram_word_dict[word] = 1
            
            
    trigram_word_dict = {}
    for i in range(len(word_list) - 2):
        word = word_list[i] + " " + word_list[i + 1] + " " + word_list[i + 2]
        try:
            trigram_word_dict[word] += 1
        except KeyError:
            trigram_word_dict[word] = 1
    
    # print(trigram_word_dict)
     

    return unigram_word_dict, bigram_word_dict, unique_words, trigram_word_dict


def __count_bigram_prob(unigram_word_dict, bigram_word_dict, unique_words) -> dict[str, int]:
    bigram_word_prob = {}
    k = 0.1

    for word in bigram_word_dict.keys():
        prev_word = word.split()[0]
        bigram_word_prob[word] = (bigram_word_dict[word] + k) / (unigram_word_dict[prev_word] + k * len(unique_words))

    return bigram_word_prob


def __count_trigram_prob(bigram_word_dict, unique_words, trigram_word_dict):
    trigram_word_prob = {}
    k = 0.1
    
    for trigram in trigram_word_dict.keys():
        prev_word = " ".join(trigram.split(" ")[:-1])
        
        if prev_word in bigram_word_dict:
            trigram_word_prob[trigram] = (trigram_word_dict[trigram] + k) / (bigram_word_dict[prev_word] + k * len(unique_words))
        else:
            trigram_word_prob[trigram] = k / (k * len(unique_words))
        
    return trigram_word_prob
    
    
    
def __trigram_transform(trigram_word_prob):
    output: dict[str, dict[str, float]] = dict()
    for word in trigram_word_prob.keys():
        s: list[str] = word.split(" ")
        w1, w2, w3 = s[0], s[1], s[2]
        prob: float = trigram_word_prob[word]

        if (w1, w2) not in output:
            d = dict()
            d[w3] = prob
            output[(w1, w2)] = d
        else:
            d = output[(w1, w2)]
            d[w3] = prob

    return output
                
                
            

def __transform(bigram_word_prob: dict[str, float]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = dict()
    for word in bigram_word_prob.keys():
        s: list[str] = word.split(" ")
        word1: str = s[0]
        word2: str = s[1]
        prob: float = bigram_word_prob[word]

        if word1 not in output:
            d = dict()
            d[word2] = prob
            output[word1] = d
        else:
            d = output[word1]
            d[word2] = prob

    return output



def __calculate_trigram_perplexity(testing_list, trigram_word_prob, bigram_word_dict, unigram_word_dict):
    total_log_prob = Decimal(0)
    N = 0  # Total number of words

    for sentence in testing_list:
        words = __parse_sentence(sentence)
        N += len(words)  # Update total word count

        for i in range(len(words) - 2):
            trigram = f"{words[i].lower()} {words[i+1].lower()} {words[i+2].lower()}"
            if trigram in trigram_word_prob:
                trigram_prob = Decimal(trigram_word_prob[trigram])
            else:
                # Apply K-Smoothing with k = 0.1
                k = Decimal(0.1)
                bigram = f"{words[i].lower()} {words[i + 1].lower()}"
                if bigram in bigram_word_dict:
                    bigram_count = Decimal(bigram_word_dict[bigram])
                    smoothed_prob = k / (bigram_count + Decimal(len(unigram_word_dict)) * k)
                else:  # If bigram doesn't exist in training data
                    smoothed_prob = k / (Decimal(len(unigram_word_dict)) * k)
                trigram_prob = smoothed_prob
            
            total_log_prob += trigram_prob.ln()  # Use natural log for probabilities

    try:
        avg_log_prob = total_log_prob / Decimal(N)
        perplexity = (Decimal(1) / avg_log_prob.exp())
    except (DivisionByZero, InvalidOperation):
        perplexity = Decimal('Infinity')

    return perplexity



def __calculate_bigram_perplexity(testing_list, bigram_word_prob, unigram_word_dict):
    total_prob = Decimal(1)
    N = 0  # Total number of words

    for sentence in testing_list:
        words = __parse_sentence(sentence)
        sent_prob = Decimal(1)  # Initialize sentence probability for each sentence
        N += len(words)  # Update total word count

        for i in range(len(words) - 1):
            bigram = words[i].lower() + " " + words[i + 1].lower()
            if bigram in bigram_word_prob:
                sent_prob *= Decimal(bigram_word_prob[bigram])
            else:
                # Apply K-Smoothing with k = 0.1
                k = Decimal(0.1)
                prev_word = words[i].lower()
                if prev_word in unigram_word_dict:
                    prev_word_count = Decimal(unigram_word_dict[prev_word])
                    smoothed_prob = k / (prev_word_count + Decimal(len(unigram_word_dict)) * k)
                else:  # If previous word doesn't exist in training data
                    smoothed_prob = k / (Decimal(len(unigram_word_dict)) * k)
                sent_prob *= smoothed_prob

        total_prob *= sent_prob

    try:
        perplexity = (Decimal(1) / total_prob) ** (Decimal(1) / N)
    except ZeroDivisionError:
        perplexity = Decimal('Infinity')

    return perplexity

def __compute_trigram_accuracy(trigram_prob, testing_set):
    trigram_word_prob = __trigram_transform(trigram_prob)
        
    total_trigrams = 0
    correct_predictions = 0
    
    for sentence in testing_set:
        words = __parse_sentence(sentence)
        
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i+1], words[i + 2]
            next_word_probs = trigram_word_prob.get((w1, w2), {})
            if next_word_probs:
                predicted_word = max(next_word_probs, key=next_word_probs.get)
                if predicted_word == w3:
                    correct_predictions += 1
            total_trigrams += 1

    return correct_predictions / total_trigrams if total_trigrams > 0 else 0



def __compute_bigram_accuracy(bigram_prob, testing_set):
    bigram_word_prob = __transform(bigram_prob)
    
    total_bigrams = 0
    correct_predictions = 0
    
    for sentence in testing_set:
        words = __parse_sentence(sentence)
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            next_word_probs = bigram_word_prob.get(w1, {})
            if next_word_probs:
                predicted_word = max(next_word_probs, key=next_word_probs.get)
                if predicted_word == w2:
                    correct_predictions += 1
            total_bigrams += 1

    return correct_predictions / total_bigrams

            

if __name__ == "__main__":
    init(True)
