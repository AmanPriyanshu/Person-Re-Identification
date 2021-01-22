import torch
from torch import nn
import numpy as np

class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()

		self.cnn1 = nn.Sequential( 
		nn.Conv2d(3, 8, kernel_size=3),
		nn.ReLU(inplace=True),
		nn.LocalResponseNorm(5),
		nn.MaxPool2d(3, stride=2),

		nn.Conv2d(8, 12, kernel_size=3),
		nn.ReLU(inplace=True),
		nn.LocalResponseNorm(5),
		nn.MaxPool2d(3, stride=2),

		nn.Conv2d(12, 8, kernel_size=3),
		nn.ReLU(inplace=True),
		nn.Conv2d(8, 4, kernel_size=3),
		nn.ReLU(inplace=True),
		nn.MaxPool2d(3, stride=2),
		)

		self.fc1 = nn.Sequential(
		nn.Linear(192, 64),
		nn.ReLU(inplace=True),
		nn.Linear(64, 32),
		nn.ReLU(inplace=True),
		nn.Linear(32,2),
		)

	def forward_once(self, x):
		output = self.cnn1(x)
		output = output.view(output.size()[0], -1)
		output = self.fc1(output)
		return output

	def forward(self, input1, input2):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)
		return output1, output2