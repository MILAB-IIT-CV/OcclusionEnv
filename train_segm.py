from model import Segmenter
import torch


if __name__=='__main__':
    model = Segmenter(8).cuda()
    img = torch.randn(8,4,512,512).cuda()
    y = model(img)
