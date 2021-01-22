import torch
from torch import nn
import numpy as np
from PIL import Image
from torchvision import transforms

from code.models.simple_siamese_network import SiameseNetwork

transform = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
								])

if __name__ == '__main__':
	img = Image.open('./dataset/train/0001_c1s1_001051_00.jpg')
	img = transform(img)
	img = torch.stack([img])
	print(img.shape)
	sn = SiameseNetwork()
	print(sn.forward_once(img).shape)