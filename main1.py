import torch
from torch.autograd import Variable
#read training sets
fxr=map(int,open('trainingSetsX.txt').read().split())
fyr=map(int,open('trainingSetsY.txt').read().split())

N=len(fxr)/400#batch size
D_in,H,D_out=400,400,400#input size, hiden layer size, output size

x=torch.FloatTensor(fxr).resize_(N,400)
y=torch.FloatTensor(fyr).resize_(N,400)
x=Variable(x,requires_grad=False)
y=Variable(y,requires_grad=False)

model=torch.nn.Sequential(
	torch.nn.Linear(D_in,H),
	torch.nn.Sigmoid(),
	
	torch.nn.Linear(H,H),
	torch.nn.Sigmoid(),

	torch.nn.Linear(H,D_out),
	torch.nn.Sigmoid(),
	)

loss_fn=torch.nn.MSELoss(size_average=False)

learning_rate=0.03
for t in range(200):
	y_pred=model(x)
	loss=loss_fn(y_pred,y)
	print(t,loss.data[0])
	model.zero_grad()
	loss.backward()
	for param in model.parameters():
		param.data-=learning_rate*param.grad.data


#read the test 
test=map(int,open('test.txt').read().split())
test=torch.FloatTensor(test).resize_(1,400)
test=Variable(test,requires_grad=False)


y_pred=model(test)
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
