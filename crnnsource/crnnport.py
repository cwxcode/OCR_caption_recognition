import sys
sys.path.insert(1, "./crnnsource")
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import keys
import os

import models.crnn as crnn

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] ="1"
    
def load_pretrained_model(model, pretrained_param):
    model_param = model.state_dict()
    assert(len(model_param) == len(pretrained_param))
    for i in model_param.keys():
        model_param[i] = pretrained_param['module.'+i]
    model.load_state_dict(model_param)
    
def crnnSource(img_path):
    model_path = './data/netCRNN_2_90000.pth'
    alphabet = keys.alphabet
    
    model = crnn.CRNN(64, 1, len(alphabet)+1, 256)
    if torch.cuda.is_available():
        model = model.cuda()
        
#    print('loading pretrained model from %s' % model_path)
    pre_model = torch.load(model_path)
    
    load_pretrained_model(model, pre_model)
    
    converter = utils.strLabelConverter(alphabet)
    
    image = Image.open(img_path).convert('L')
#    transformer = dataset.resizeNormalize((200, 64))
    #改变图像尺寸
    scale = image.size[1]*1.0 / 64
    w = image.size[0] / scale
    w = int(w)

    transformer = dataset.resizeNormalize((w, 64))

#    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    
    model.eval()
    preds = model(image)
    
    _, preds = preds.max(2, keepdim=True)
    preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
