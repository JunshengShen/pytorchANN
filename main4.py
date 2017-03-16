import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
#read training sets
fxr=map(int,open('x50.txt').read().split())
fxr2=map(int,open('circle.txt').read().split())
fyr=map(int,open('y50.txt').read().split())

N=len(fxr)/10000#batch size


x=torch.FloatTensor(fxr).resize_(N,1,100,100)
x2=torch.FloatTensor(fxr2).resize_(1,100,100)
x2=torch.cat((x2,x2,x2,x2,x2,x2,x2,x2,x2,x2),0)
x2.resize_(N,1,100,100)
X=torch.cat((x,x2),1)
X.resize_(N,1,100,200)
print X

y=torch.FloatTensor(fyr).resize_(N,1,100,100)
X=Variable(X,requires_grad=False)

y=Variable(y,requires_grad=False)
#a1=Variable(torch.randn(N,81,2,2).type(torch.FloatTensor),requires_grad=True)
#a2=Variable(torch.randn(N,27,4,4).type(torch.FloatTensor),requires_grad=True)
#a3=Variable(torch.randn(N,9,9,9).type(torch.FloatTensor),requires_grad=True)
#a4=Variable(torch.randn(N,3,19,19).type(torch.FloatTensor),requires_grad=True)
#print a1
class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.conv1 = nn.Conv2d(1,2,(10,20),stride=(2,4))
		self.conv2 = nn.Conv2d(2,4,10)
		self.conv3 = nn.Conv2d(4,8,10)
		self.conv4 = nn.Conv2d(8,16,10)
		self.conv5 = nn.Conv2d(16,32,10)
		self.bn1 = nn.BatchNorm2d(2)
		self.bn2 = nn.BatchNorm2d(4)
		self.bn3 = nn.BatchNorm2d(8)
		self.bn4 = nn.BatchNorm2d(16)
		self.bn5 = nn.BatchNorm2d(32)
		self.convT1 = nn.ConvTranspose2d(32,16,10)
		self.convT2 = nn.ConvTranspose2d(16,8,10)
		self.convT3 = nn.ConvTranspose2d(8,4,10)
		self.convT4 = nn.ConvTranspose2d(4,2,10)
		self.convT5 = nn.ConvTranspose2d(2,1,10,stride=2)
		self.bn6 = nn.BatchNorm2d(16)
		self.bn7 = nn.BatchNorm2d(8)
		self.bn8 = nn.BatchNorm2d(4)
		self.bn9 = nn.BatchNorm2d(2)	
		self.bn10 = nn.BatchNorm2d(1)



	def forward(self,x):
		encode1 = self.conv1(x)
		encode1 = self.bn1(encode1)
		encode2 = self.conv2(encode1)
		encode2 = self.bn2(encode2)
		encode3 = self.conv3(encode2)
		encode3 = self.bn3(encode3)
		encode4 = self.conv4(encode3)
		encode4 = self.bn4(encode4)
		encode5 = self.conv5(encode4)
		encode5 = self.bn5(encode5)
		
		#print encode5
		#print encode4
		decode1 = self.convT1(encode5) #+ a1*encode4
		decode1 = self.bn6(decode1)
		decode2 = self.convT2(decode1) #+ a2*encode3
		decode2 = self.bn7(decode2)
		decode3 = self.convT3(decode2) #+ a3*encode2
		decode3 = self.bn8(decode3)
		decode4 = self.convT4(decode3) #+ a4*encode1
		decode4 = self.bn9(decode4)
		decode5 = self.convT5(decode4)
		decode5 = self.bn10(decode5)
		#print decode1
		return decode5




model=Model()

print model

loss_fn=torch.nn.MSELoss(size_average=False)
model=torch.load('modelSaved100_50bn_5')
learning_rate=0.000000003
for t in range(0):
	y_pred=model(X)
	loss=loss_fn(y_pred,y)
	print(t,loss.data[0])
	model.zero_grad()
	#a1.grad.data.zero_()
	#a2.grad.data.zero_()
	#a3.grad.data.zero_()
	#a4.grad.data.zero_()
	loss.backward()
	for param in model.parameters():
		param.data-=learning_rate*param.grad.data
		#print a1.grad.data
		#print a1.grad
		#a1.data=learning_rate*a1.grad.data
		#a2.data=learning_rate*a2.grad.data
		#a3.data=learning_rate*a3.grad.data
		#a4.data=learning_rate*a4.grad.data
#print a1
#print a2
#print a3
#print a4

#torch.save(model,'modelSaved100_50bn_5')
#model=torch.load('modelSaved100ReLU_2')
#read the test 
test=map(int,open('outfile.txt').read().split())
test1=torch.FloatTensor(test).resize_(1,1,100,100)
#test1=Variable(test1,requires_grad=False)
#testr1=map(int,open('x.txt').read().split())
testr2=map(int,open('circle.txt').read().split())
#test=torch.FloatTensor(testr1).resize_(1,100,100)

test2=torch.FloatTensor(testr2).resize_(1,1,100,100)

test=torch.cat((test1,test2),3)
#test.resize_(1,1,100,100)


#X=Variable(test,requires_grad=False)


#print X

y_pred=model(X)

y_pred=torch.FloatTensor(y_pred.data).resize_(1,10000)

#y_pred=x.mm(w1).clamp(min=0).mm(w2)
a=[]
#print y_pred[0][1]
for i in range(10000):
	a.append(y_pred[0][i])
pre=str(a)

pre=pre.replace("[","")
pre=pre.replace("]","")+"\n"


f=open("a.txt","w")
f.write(pre)
f.close()
