import nltk


s = "leralmin-tor lucy, vi had vesh' further saddened k' wuh signora’s ri-gishu accent. ek'wak."




punctuation = "!\"#$%&()*+, ./:;<=>?@[]^_\`{|}~"
def parseSentence(sentence: str):
    sentence = sentence.replace('’', '').replace("\n", ' ')
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
print(parseSentence(s))

