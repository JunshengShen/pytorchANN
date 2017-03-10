import torch
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


model=torch.nn.Sequential(
torch.nn.Conv2d(1,3,(2,4),stride=(1,2)),
#torch.nn.Sigmoid(),
torch.nn.Conv2d(3,9,3,stride=2),
#torch.nn.Sigmoid(),
torch.nn.Conv2d(9,27,3,stride=2),
torch.nn.Conv2d(27,81,3),
torch.nn.Conv2d(81,243,2),
#torch.nn.Sigmoid(),
#torch.nn.Conv2d(8,16,3,stride=2),

#torch.nn.Conv2d(16,32,3,stride=2),
#torch.nn.Sigmoid(),
#torch.nn.Conv2d(32,64,3,stride=2),
#torch.nn.Sigmoid(),
#orch.nn.ConvTranspose2d(64,32,3,stride=2),
#torch.nn.Sigmoid(),
#torch.nn.ConvTranspose2d(32,16,3,stride=2),
#torch.nn.Sigmoid(),
#torch.nn.ConvTranspose2d(16,8,3,stride=2),
#torch.nn.Sigmoid(),
torch.nn.ConvTranspose2d(243,81,2),
torch.nn.ConvTranspose2d(81,27,3),
torch.nn.ConvTranspose2d(27,9,3,stride=2),
#torch.nn.Sigmoid(),
torch.nn.ConvTranspose2d(9,3,3,stride=2),

torch.nn.ConvTranspose2d(3,1,2),
#torch.nn.Sigmoid(),
)


print model

loss_fn=torch.nn.MSELoss(size_average=False)
model=torch.load('modelSavedStride2Tur3_1')
learning_rate=0.0001
for t in range(	10):
	y_pred=model(X)
	loss=loss_fn(y_pred,y)
	print(t,loss.data[0])
	model.zero_grad()
	loss.backward()
	for param in model.parameters():
		param.data-=learning_rate*param.grad.data

#torch.save(model,'modelSavedStride2Tur3_1')
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
