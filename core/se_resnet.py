import torch
import torch.nn as nn
import torch.nn.functional as F
class ChannelAttention(nn.Module):
	def __init__(self, channel, reduction=4):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y.expand_as(x)
class SpatialAttention(nn.Module):
	def __init__(self, channel,kernel_size=3):
		super(SpatialAttention, self).__init__()

		# assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
		# padding = 3 if kernel_size == 7 else 1

		self.conv1 = nn.Conv2d(2, channel, kernel_size, padding = 1, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = torch.mean(x,dim = 1,keepdim=True)
		max_out, _ = torch.max(x,dim = 1,keepdim=True)
		x = torch.cat([avg_out, max_out], dim=1)
		x = self.conv1(x)

		return self.sigmoid(x)
class resnet_block(nn.Module):
	def __init__(self, inchannel, outchannel, stride=1):
		super(resnet_block, self).__init__()
		self.left = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel),
			nn.ReLU(inplace=True),
			nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel),
			ChannelAttention(outchannel),
			SpatialAttention(outchannel)
		)
		self.shortcut = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel),
			nn.ReLU(inplace=True),
			nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel),
		)
	def forward(self,x):
		# print(x.shape)
		out = self.shortcut(x)
		# print(out.shape,self.left(x).shape)
		out = out+self.left(x)
		return nn.functional.relu(out)
class se_resnet(nn.Module):
	def __init__(self, imgsz,num_classes):
		super(se_resnet, self).__init__()
		# self.inchannel = 64
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU()
		)
		self.layer1 = resnet_block(16,16)
		self.layer2 = resnet_block(16,32)
		self.layer3 = resnet_block(32,32)
		self.layer4 = resnet_block(32,64)
		self.pool = nn.MaxPool2d(2,2)
		self.fc = nn.Linear(64*int(imgsz*imgsz/64),num_classes)
		# self.fc1 = nn.Linear(1024,16)
		# self.fc2 = nn.Linear(16,num_classes)
	def forward(self,x):
		out = self.conv1(x)
		out = self.pool(out)
		out = self.layer1(out)
		# out = self.pool(out)
		out = self.layer2(out)
		out = self.pool(out)
		out = self.layer3(out)
		# out = self.pool(out)
		out = self.layer4(out)
		out = torch.flatten(self.pool(out),1)
		out = self.fc(out)
		return nn.functional.softmax(out, dim=1)


if __name__ =="__main__" :
	se = se_resnet(64,3)
	a = torch.randn(1, 3, 64, 64)
	print(se(a))