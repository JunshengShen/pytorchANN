import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
#read training sets
fxr=map(int,open('trainingSetsX.txt').read().split())
fxr2=map(int,open('trainingSetsX2.txt').read().split())
fyr=map(int,open('trainingSetsY.txt').read().split())

N=len(fxr)/400#batch size
D_in,H,D_out=400,400,400#input size, hiden layer size, output size

x=torch.FloatTensor(fxr).resize_(N,20,20)
x2=torch.FloatTensor(fxr2).resize_(N,20,20)
X=torch.cat((x,x2),1)
X.resize_(N,1,20,40)
y=torch.FloatTensor(fyr).resize_(N,1,20,20)
X=Variable(X,requires_grad=False)
y=Variable(y,requires_grad=False)
a1=Variable(torch.randn(N,81,2,2).type(torch.FloatTensor),requires_grad=True)
a2=Variable(torch.randn(N,27,4,4).type(torch.FloatTensor),requires_grad=True)
a3=Variable(torch.randn(N,9,9,9).type(torch.FloatTensor),requires_grad=True)
a4=Variable(torch.randn(N,3,19,19).type(torch.FloatTensor),requires_grad=True)
print a1
class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.conv1 = nn.Conv2d(1,3,(2,4),stride=(1,2))
		self.conv2 = nn.Conv2d(3,9,3,stride=2)
		self.conv3 = nn.Conv2d(9,27,3,stride=2)
		self.conv4 = nn.Conv2d(27,81,3)
		self.conv5 = nn.Conv2d(81,243,2)
		
		self.convT1 = nn.ConvTranspose2d(243,81,2)
		self.convT2 = nn.ConvTranspose2d(81,27,3)
		self.convT3 = nn.ConvTranspose2d(27,9,3,stride=2)
		self.convT4 = nn.ConvTranspose2d(9,3,3,stride=2)
		self.convT5 = nn.ConvTranspose2d(3,1,2)
		
	def forward(self,x):
		encode1 = self.conv1(x)
		encode2 = self.conv2(encode1)
		encode3 = self.conv3(encode2)
		encode4 = self.conv4(encode3)
		encode5 = self.conv5(encode4)
		
		
		#print encode4
		decode1 = self.convT1(encode5) + a1*encode4
		decode2 = self.convT2(decode1) + a2*encode3
		decode3 = self.convT3(decode2) + a3*encode2
		decode4 = self.convT4(decode3) + a4*encode1
		decode5 = self.convT5(decode4)
		#print decode1
		return decode5




model=Model()

print model

loss_fn=torch.nn.MSELoss(size_average=False)
#model=torch.load('modelSaved7')
learning_rate=0.0001
for t in range(10000):
	y_pred=model(X)
	loss=loss_fn(y_pred,y)
	print(t,loss.data[0])
	model.zero_grad()
	a1.grad.data.zero_()
	a2.grad.data.zero_()
	a3.grad.data.zero_()
	a4.grad.data.zero_()
	loss.backward()
	for param in model.parameters():
		param.data-=learning_rate*param.grad.data
		#print a1.grad.data
		#print a1.grad
		a1.data=learning_rate*a1.grad.data
		a2.data=learning_rate*a2.grad.data
		a3.data=learning_rate*a3.grad.data
		a4.data=learning_rate*a4.grad.data
print a1
print a2
print a3
print a4

#torch.save(model,'modelSaved8')
#model=torch.load('savetest')
#read the test 
#test=map(int,open('test.txt').read().split())
#test=torch.FloatTensor(test).resize_(1,1,20,20)
#test=Variable(test,requires_grad=False)
testr1=map(int,open('testx1.txt').read().split())
testr2=map(int,open('testx2.txt').read().split())
test1=torch.FloatTensor(testr1).resize_(N,20,20)

test2=torch.FloatTensor(testr2).resize_(N,20,20)

test=torch.cat((test1,test2),1)
test.resize_(N,1,20,40)


X=Variable(test,requires_grad=False)




y_pred=model(X)

y_pred=torch.FloatTensor(y_pred.data).resize_(1,400)

#y_pred=x.mm(w1).clamp(min=0).mm(w2)
a=[]
for i in y_pred:
	a.append(i)
pre=str(a)

pre=pre.replace("[","")
pre=pre.replace("]","")+"\n"


f=open("a.txt","w")
f.write(pre)
f.write(pre)
f.close()
