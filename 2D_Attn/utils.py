#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections.abc
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        
        self.alphabet = alphabet
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char.strip()] = i + 1
        self.max_val = 1
        ctc_blank = '~'
        self.alphabet.insert(0, ctc_blank)
        #print(self.dict)
        #print(self.alphabet)

    def encode(self, text):
        """Support batch or single str.
        """
        length = []
        result = []
        for item in text:
            # item = item.decode()
            length.append(len(item))
            r = []
            for char in item:
            	#newly added
                if char != '\u200c':
                    index = self.dict[char]
                r.append(index)
            result.append(r)

        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)

        result_temp = []
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)

        text = result_temp
        #print(text)
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            #print(t)
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), self.max_val)
            if raw:
                return ''.join([self.alphabet[i].strip() for i in t])
            else:
                char_list = []
                try:
                    for i in range(length):
                        if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                            char_list.append(self.alphabet[t[i]].strip())
                except:
                    print(t)

                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index : index+l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]','[UNK]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        #self.alphabet = self.character
        #self.max_val = 1
        #ctc_blank = '~'
        #self.alphabet.insert(0, ctc_blank)

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char.strip()] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        #print(self.dict)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            #text = [self.dict[char] for char in text]
            t=[]
            for char in text:
                if char not in self.dict.keys():
                    print('char: ',char)
                    t.append(self.dict['[UNK]'])
                else:
                    t.append(self.dict[char])
            text = t
            #text = [self.dict[char] for char in text]

            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, t, length, raw=False):
        """ convert text-index into text-label. """
        texts = []
#        print(text_index.shape)
#        print(length.shape)
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in t[index,:]])
#            print(text)
            texts.append(text)
        return texts
#        if length.numel() == 1:
#             length = length[0]
#             #print(t)
#             assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), self.max_val)
#             if raw:
#                 return ''.join([self.alphabet[i].strip() for i in t])
#             else:
#                 char_list = []
#                 try:
#                     for i in range(length):
#                         if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
#                             char_list.append(self.alphabet[t[i]].strip())
#                 except:
#                     print(t)
#
#                 return ''.join(char_list)
#        else:
#             # batch mode
#             assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
#             texts = []
#             index = 0
#             for i in range(length.numel()):
#                 l = length[i]
#                 texts.append(
#                     self.decode(
#                         t[index : index+l], torch.IntTensor([l]), raw=raw))
#                 index += l
#             return text      

class AttnLabelConverter_withCTC(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]','[UNK]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character
        
        self.alphabet = self.character
        self.max_val = 1
        ctc_blank = '~'
        self.alphabet.insert(0, ctc_blank)

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char.strip()] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        #print(self.dict)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            #text = [self.dict[char] for char in text]
            t=[]
            for char in text:
            #for i in range(min(26,len(text))):
                if char not in self.dict.keys():
                    print('char: ',char)
                    t.append(self.dict['[UNK]'])
                else:
                    t.append(self.dict[char])
            text = t
            
            #text = [self.dict[char] for char in text]

            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, t, length, raw=False):
        """ convert text-index into text-label. """
#        texts = []
#        print(text_index.shape)
#        print(length.shape)
#        for index, l in enumerate(length):
#            text = ''.join([self.character[i] for i in text_index])
#            print(text)
#            texts.append(text)
#        return texts
        if length.numel() == 1:
             length = length[0]
             #print(t)
             assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), self.max_val)
             if raw:
                 return ''.join([self.alphabet[i].strip() for i in t])
             else:
                 char_list = []
                 try:
                     for i in range(length):
                         if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                             char_list.append(self.alphabet[t[i]].strip())
                 except:
                     print(t)

                 return ''.join(char_list)
        else:
             # batch mode
             assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
             texts = []
             index = 0
             for i in range(length.numel()):
                 l = length[i]
                 texts.append(
                     self.decode(
                         t[index : index+l], torch.IntTensor([l]), raw=raw))
                 index += l
             return texts
