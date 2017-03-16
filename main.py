import torch
import math
from torch.autograd import Variable
#read training sets
fx=open('trainingSetsX.txt')
fy=open('trainingSetsY.txt')
fxr=map(int,fx.read().split())
fyr=map(int,fy.read().split())

#dtype=torch.FloatTensor 
dtype=torch.cuda.FloatTensor #for GPU

N=len(fxr)/400#batch size
D_in,H,D_out=400,400,400#input size, hiden layer size, output size
x=torch.FloatTensor(fxr).resize_(N,400)
y=torch.FloatTensor(fyr).resize_(N,400)
x=Variable(x,requires_grad=False)
y=Variable(y,requires_grad=False)
w1=Variable((torch.randn(D_in,H)/100).type(dtype),requires_grad=True)    #hidden layer 1
w2=Variable((torch.randn(H,H)/100).type(dtype),requires_grad=True)       #hidden layer 2
w3=Variable((torch.randn(H,D_out)/100).type(dtype),requires_grad=True)   #output layer 
learning_rate=0.03
lamda=0.1
for t in range(200):
	z1= x.mm(w1)
	a1=1/(1+math.e**(-z1))

	z2=a1.mm(w2)
	a2=1/(1+math.e**(-z2))

	z3=a2.mm(w3)
	y_pred=1/(1+math.e**(-z3))

	
	#y_pred=x.mm(w1).clamp(min=0).mm(w2)
	loss = (y_pred - y).pow(2).sum()+lamda*(w1.pow(2).sum()+w2.pow(2).sum()+w3.pow(2).sum())
	print(t, loss.data[0])
	w1.grad.data.zero_()
	w2.grad.data.zero_()
	w3.grad.data.zero_()

	loss.backward()
	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data
	w3.data -= learning_rate * w3.grad.data
	

#read the test 
test=map(int,open('test.txt').read().split())
test=torch.FloatTensor(test).resize_(1,400)
test=Variable(test,requires_grad=False)

z1= test.mm(w1)
a1=1/(1+math.e**(-z1))

z2=a1.mm(w2)
a2=1/(1+math.e**(-z2))

z3=a2.mm(w3)
y_pred=1/(1+math.e**(-z3))
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
