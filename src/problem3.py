import math
from decimal import Decimal

import nltk

INPUT_FILE_PATH = "doyle_Bohemia.txt"
OUTPUT_PROB_PATH = "smooth_probs.txt"
OUTPUT_EVAL_PATH = "smoothed_eval.txt"
ENCODING = "utf8"



# Read file and break into sentences
with open(INPUT_FILE_PATH, "r", encoding=ENCODING) as input_file:
    sentence_list = nltk.sent_tokenize(input_file.read())

# Split 80% of sentences for training
training_list = sentence_list[: int(len(sentence_list) * 0.8)]

# Convert each sentence to lower-case tokens
all_words_list = []
for sentence in training_list:
    words = nltk.word_tokenize(sentence)
    for i in range(len(words)):
        words[i] = words[i].lower()
    all_words_list.extend(words)

# Count frequency of individual words
# And add words to a set to create a list of unique words
unique_words = set()
unigram_word_dict = {}
for word in all_words_list:
    unique_words.add(word)
    try:
        unigram_word_dict[word] += 1
    except KeyError:
        unigram_word_dict[word] = 1

# Count frequency of pair words
bigram_word_dict = {}
for i in range(len(all_words_list) - 1):
    word = all_words_list[i] + " " + all_words_list[i + 1]
    try:
        bigram_word_dict[word] += 1
    except KeyError:
        bigram_word_dict[word] = 1

# Using the individual words, pair words, and unique words,
# calculate the bigram probability of each pair word with smoothing
# with k = 0.1
bigram_word_prob = {}
k = 0.1
for word in bigram_word_dict.keys():
    prev_word = word.split()[0]
    bigram_word_prob[word] = (bigram_word_dict[word] + k) / (unigram_word_dict[prev_word] + k * len(unique_words))

# Format the bigram probabilities for output file
output_string = ""
for word in bigram_word_prob.keys():
    s = word.split(" ")
    output_string += "p(" + s[1] + "|" + s[0] + ") = " + str(bigram_word_prob[word]) + "\n"

with open(OUTPUT_PROB_PATH, "w", encoding=ENCODING) as output_file:
    output_file.write(output_string)

print("Finished reading '" + INPUT_FILE_PATH + "' and dumping to probabilities to file '" + OUTPUT_PROB_PATH + "'")

#
#   STEP 3: Finding joint distribution
#

# Use remaining 20% of sentences for testing
testing_list = sentence_list[int(len(sentence_list) * 0.8):]
total_prob = Decimal(1)

eval_output_string = ""
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

    total_prob *= Decimal(sent_prob)
    eval_output_string += "p(" + sentence.replace("\n", " ") + ") = " + str(sent_prob) + "\n"

try:
    perplexity = (1 / total_prob) ** Decimal(1 / len(testing_list))
except ZeroDivisionError:
    perplexity = "INF"

print("Total Prob: " + str(total_prob))
print("Perplexity: " + str(perplexity))
# Write to output file
with open(OUTPUT_EVAL_PATH, "w", encoding=ENCODING) as eval_output_file:
    eval_output_file.write(eval_output_string)
