import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3, 2]  #kernel 核大小
        ps = [1, 1, 1, 1, 1, 1, 1, 0]  #padding 填充大小
        ss = [1, 1, 1, 1, 1, 1, 1, 1]  #stride 步长
        nm = [64, 128, 256, 256, 256, 512, 512, 512]  #卷积核个数

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):  #定义relu方式卷积
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)  # 1x64x200  1表示特征图数量
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x100
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x50
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 256x8x25
        convRelu(4)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2,1), (0,1)))  # 256x4x26 新增的
        convRelu(5, True)
        convRelu(6)
        cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x27
        convRelu(7, True)  # 512x1x26

        self.cnn = cnn
        self.rnn = nn.Sequential(  #两层RNN，隐藏层数量都是256
            BidirectionalLSTM(512, nh, nh),  #输入是512
            BidirectionalLSTM(nh, nh, nclass))  #输出是类别数

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
#	print("b: %d, c: %d, h: %d, w: %d" %(b, c, h, w))
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  #将输入张量形状中的1去除并返回
        conv = conv.permute(2, 0, 1)  # [w, b, c]  对任意高维矩阵进行转置

        # rnn features
        output = self.rnn(conv)

        return output
