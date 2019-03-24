from nltk import PunktSentenceTokenizer, WordPunctTokenizer
from collections import Counter

vocab_size = 1000

sentTokenier = PunktSentenceTokenizer()
wordTokenizer = WordPunctTokenizer()

filename = 'data/formatted_movie_lines.txt'
string = open(filename, mode='r', encoding='utf8').read()
string = string.replace("'t", "")
string = string.replace("'s", "")

words = wordTokenizer.tokenize(string)
sentences = set(sentTokenier.tokenize(string))

vocab = Counter(words).most_common(vocab_size)
dict = Counter(vocab)
sentences = [wordTokenizer.tokenize(sentence) for sentence in sentences]

new_sentences = []
with open("lines.txt", mode='w', encoding='utf8') as file:
    for sentence in sentences:
        write = True
        for word in sentence:
            if word in dict.keys():
                write = False
                break
        if write:
            file.writelines(" ".join(sentence) + "\n")
            new_sentences.append(sentence)

print("Length of filtered sentences : ", len(new_sentences), len(sentences))
