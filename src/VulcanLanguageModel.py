from decimal import Decimal
import os
import nltk
import string
nltk.download('punkt')

TEXTS_PATH = './data/'

formatted_bigram_prob = None


def init():
    global formatted_bigram_prob
    if formatted_bigram_prob != None:
        print("Vulcan Language Model is already initialized...")
        return
    
    
    print("Initializing Vulcan Language Model...")
    text_docs, test_sentences = __load_docs()
    
    print("Words in training set: " + str(len(text_docs)))
    print("Counting frequencies...")
    
    unigram_word_dict, bigram_word_dict, unique_words = __count_freq(text_docs)
    
    print("Counting probabilities...")
    bigram_word_prob = __count_bigram_prob(unigram_word_dict, bigram_word_dict, unique_words)
    
    formatted_bigram_prob = __transform(bigram_word_prob)
    print("Init complete")
    
    perplexity = calculatePerplexity(test_sentences, bigram_word_prob, unigram_word_dict)
    print("perplexity:", perplexity)
    
    
def predict(word: str) -> tuple[str, str, str]:
    if formatted_bigram_prob == None:
        raise Exception("predict() was called before init().")
    
    def getWord(arr: list, index: int):
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
        if ((tup[0] in string.punctuation) == False) and ((tup[0] in blacklist) == False):
            results.append(tup)

    
    return (getWord(results, 0), getWord(results, 1), getWord(results, 2))
 
    
    
    
blacklist = '”’‘“ …'
punctuation = "!\"#$%&()*+, ./:;<=>?@[]^_\`{|}~"
def __parseSentence(sentence: str):
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
        words = __parseSentence(sentence)
        for i in range(len(words)):
            words[i] = words[i].lower()
        all_words.extend(words)
    
    
    return all_words, testing_set_sentences


def __count_freq(word_list: list[str]):
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
            
    return unigram_word_dict, bigram_word_dict, unique_words


def __count_bigram_prob(unigram_word_dict, bigram_word_dict, unique_words):
    bigram_word_prob = {}
    k = 0.1
    
    
    for word in bigram_word_dict.keys():
        prev_word = word.split()[0]
        bigram_word_prob[word] = (bigram_word_dict[word] + k) / (unigram_word_dict[prev_word] + k * len(unique_words))

    
    return bigram_word_prob


def __transform(bigram_word_prob: dict[str, float]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = dict()
    for word in bigram_word_prob.keys():
        s: list[str] = word.split(" ")
        word1: str = s[0]
        word2: str = s[1]
        prob: float = bigram_word_prob[word]
        
        
        if not word1 in output:
            d = dict()
            d[word2] = prob
            output[word1] = d
        else:
            d = output[word1]
            d[word2] = prob
    
    return output
        


    


def calculatePerplexity(testing_list, bigram_word_prob, unigram_word_dict):
    total_prob = Decimal(1)
    

    for sentence in testing_list:
        sent_prob = 1
        words = nltk.word_tokenize(sentence)
        for i in range(len(words) - 1):
            try:
                bigram = words[i].lower() + " " + words[i + 1].lower()
                sent_prob *= bigram_word_prob[bigram]
            except KeyError:
                # Apply K-Smoothing with k = 0.1
                k = 0.1
                try:
                    prev_word_count = unigram_word_dict[words[i].lower()]
                    sent_prob *= k / (prev_word_count + len(unigram_word_dict) * k)
                except KeyError:  # If prev word doesn't exist in training data
                    sent_prob *= k / (len(unigram_word_dict) * k)


    try:
        perplexity = (1 / total_prob) ** Decimal(1 / len(testing_list))
    except ZeroDivisionError:
        perplexity = "INF"
        
    return perplexity


if __name__ == "__main__":
    init()
