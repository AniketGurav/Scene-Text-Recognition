import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os
from nltk.metrics.distance import edit_distance
import argparse
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
#import lmdb
import sys
from PIL import Image
import numpy as np
import io
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.attn_model as am
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model to test on')
parser.add_argument('--test_data', required=True, help='path to testing dataset')
parser.add_argument('--lexicon', required=True, help='path to the lexicon file')
#parser.add_argument('--type', required=True, help='Choose CRNN or STAR-Net')
opt = parser.parse_args()
print(opt)

class loadDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.imagePathList = []
        self.labelList = []
        with open(os.path.join(root,'label.txt'), 'r', encoding='utf-8') as file:
            for item in file.readlines():
                image, word = item.rstrip().split()	
                #image = image.split('/')	
                image = os.path.join(root, image)
                self.imagePathList.append(image)
                self.labelList.append(word)
        
        self.nSamples = len(self.labelList)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= self.nSamples, 'index range error'

        try:
            img = Image.open(self.imagePathList[index])
            img = img.convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        label = self.labelList[index]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


model_path = opt.model
lexicon_filename = opt.lexicon
p = open(lexicon_filename, 'r').readlines()
alphabet = p

converter = utils.AttnLabelConverter(alphabet)

#nclass = len(p) + 1
nclass = len(converter.character)
print(nclass)

model = am.Attn_model(32, 100, 1, nclass, 256)

if torch.cuda.is_available():
    model = model.cuda()

model.apply(weights_init)
model_dict = model.state_dict()
checkpoint = torch.load(model_path)
for key in list(checkpoint.keys()):
  if 'module.' in key:
    checkpoint[key.replace('module.', '')] = checkpoint[key]
    del checkpoint[key]

model_dict.update(checkpoint)
model.load_state_dict(checkpoint)

vocab = []
for i in alphabet:
  vocab.append(i.strip())

image = torch.FloatTensor(32, 3, 32, 32)
text = torch.LongTensor(32 * 5)
length = torch.LongTensor(32)

model = model.cuda()
moedl = torch.nn.DataParallel(model, device_ids=range(1))
image = image.cuda()
#criterion = criterion.cuda()


image = Variable(image)
text = Variable(text)
length = Variable(length)
loss_avg = utils.averager()

def testeval(img,lab):
 
    transformer = dataset.resizeNormalize((100, 32))
    image = transformer(img)
    if torch.cuda.is_available():
        image = image.cuda()

    image = image.view(1, *image.size())
    image = Variable(image)

    #new
    t, l = converter.encode(lab)
    
    model.eval()
    preds = model(image,t)
    #new
    #preds = preds.permute(1,0,2)
    
    _, preds = preds.max(2)
    preds = preds.squeeze(1)
    #new
    preds = preds.transpose(1, 0).contiguous().view(-1)
    
    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    #raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return img, sim_pred

def test_batch(model, test_dataset):
    
    print('Start test')
    #new
    batchSize = 32
    batch_max_length = 25
    
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=batchSize, num_workers=1)
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    norm_ED = 0
    #max_iter = min(max_iter, len(data_loader))
    max_iter = len(data_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        
        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)


        #if batch_size != opt.batchSize:
        #	continue
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        #print('Text size'+t.size())
        utils.loadData(text, t)
        utils.loadData(length, l)

        #preds = model(image, t[:, :-1],is_train=False)  # align with Attention.forward
        preds = model(image, text_for_pred, is_train=False)
        preds = preds[:, :t.shape[1] - 1, :]
        #print(preds.shape)
        #preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        target = t[:, 1:]  # without [GO] Symbol
        #cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

        #loss_avg.add(cost)

        _, preds = preds.max(2)
        #preds = preds.squeeze(1)
        #preds = preds.transpose(1, 0).contiguous().view(-1)
        #sim_preds = converter.decode(preds, preds_size.data)
        sim_preds = converter.decode(preds, length_for_pred)
        cpu_texts = converter.decode(t[:, 1:], l)
        
        for pred, target in zip(sim_preds, cpu_texts):
            target = target.strip()
            gt = target.strip()
            gt = gt[:gt.find('[s]')]
            pred = pred[:pred.find('[s]')]

            if pred == gt:
                n_correct += 1
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)
    #raw_preds = converter.decode(preds.data, preds_size.data)[:opt.n_test_disp]
    for pred, gt in zip(sim_preds, cpu_texts):
        pred = pred[:pred.find('[s]')]
        pred = ''.join([i for i in pred])
        gt = gt[:gt.find('[s]')]
        gt = ''.join([i for i in gt])
        
        pred = pred.replace('\n','')
        gt = gt.replace('\n','')

        print('pred: %-20s, gt: %-20s' % (pred.strip(), gt.strip()))
    print("Samples Correctly recognised = " + str(n_correct))
    #print(max_iter,batch_size)
    accuracy = n_correct / float(max_iter * batchSize)
    crr = norm_ED / float(max_iter * batchSize)
    #lossval = loss_avg.val()
    #print('Wcc accuracy: %f, crr: %f' % (accuracy, crr))
    return accuracy, crr

#lmdb_data = opt.test_data
test_dataset = dataset.loadDataset(opt.test_data, transform=dataset.resizeNormalize((100, 32))) 

wrr, crr = test_batch(model,test_dataset)
print("FINAL RESULTS")
#print("Average Loss :" + str(loss))
print("Word Recognition Rate :" + str(wrr))
print("Character Recognition Rate :" + str(crr))
