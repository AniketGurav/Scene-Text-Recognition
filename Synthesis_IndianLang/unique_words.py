import numpy as np
import codecs, sys

vocabFile = codecs.open(sys.argv[1],'r',encoding='utf8')
words = vocabFile.read().split()
words = np.array(words)

print("Total words: ", len(words))
print("Unique words:",len(np.unique(words)))
#print(words[4816])
#print(words[4875])
#print(words[117])
#print(words[8])
# print(''.join(words).split())
# print(words[4816][-1]=='\u200d')
# print(words[195009][-1].encode('ascii'))
