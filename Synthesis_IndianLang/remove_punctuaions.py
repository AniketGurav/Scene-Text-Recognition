import numpy as np
import codecs, sys

vocabFile = codecs.open(sys.argv[1],'r',encoding='utf8')
words = vocabFile.read().split()
vocabFile.close()

file = open('Mal_val1.txt', 'w')
for w in words:
    if ~(w.isalpha()):
        new_w = ''
        print(w)
        for i in range(len(w)):
            if w[i] not in "[-)(#/%$^&@;:<>`+=~|.*—! –?…,\\]abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'}{\"" and w[i] not in ['\u200d','\u201d','\u201c','\u2018','\u2019','\u200c','\u200b']:
                new_w += w[i]
        print(new_w)
        w = new_w
    if len(w):
        file.write(w+'\n')

# words = words.split(' ')
# words = ''.join(words)
# print(words)
print("Total words: ", len(words))
print("Unique words:",len(np.unique(words)))

# file = open('rendered_vocab.txt', 'w')
# words = ''.join(words)
# file.write(words)
file.close()
