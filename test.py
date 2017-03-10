import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.autograd import Variable
#m=nn.Conv2d(1,2,2)
#input=torch.autograd.Variable(torch.randn(10,1,5,5))
#output=m(input)
#print input
#print output
#m=nn.ConvTranspose2d(3,1,3,stride=2)
#input=torch.autograd.Variable(torch.randn(1,3,3,3))
#output=m(input)
#print output

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
print X

model=torch.nn.Sequential(
torch.nn.Conv2d(1,3,(2,4),stride=(1,2)),
torch.nn.Conv2d(3,9,3),
torch.nn.Conv2d(9,27,3),
torch.nn.Conv2d(27,81,3),
torch.nn.ConvTranspose2d(81,27,3),
torch.nn.ConvTranspose2d(27,9,3),
torch.nn.ConvTranspose2d(9,3,3),
torch.nn.ConvTranspose2d(3,1,2),
)

y_pred=model(X)
print y_pred
