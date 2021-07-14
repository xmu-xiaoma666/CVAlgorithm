import torch
from torch import nn
from torch.optim import optimizer


"""
    定义输入的数据
"""
x0=torch.randn(100,2)
x1=-torch.randn(100,2)
y0=torch.zeros(100)
y1=torch.ones(100)
data=torch.cat([x0,x1],dim=0).type(torch.FloatTensor) #200,2
label=torch.cat([y0,y1],dim=0).type(torch.LongTensor) #200

"""
    定义模型
"""
class Net(nn.Module):
    def __init__(self,in_ch,hid_ch,out_ch):
        super().__init__()
        self.fc1=nn.Linear(in_ch,hid_ch)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hid_ch,out_ch)
        self.softmax=nn.Softmax(-1)
    
    def forward(self,x):
        return self.softmax(self.fc2(self.relu(self.fc1(x))))


"""
    实例化optimizer，model，loss_func
"""
net=Net(2,10,2)
opti=torch.optim.Adam(net.parameters(),lr=1e-3)
loss_func=nn.CrossEntropyLoss()

for i in range(1000):
    out=net(data)
    loss=loss_func(out,label)

    opti.zero_grad() # clear gradients for next train
    loss.backward() # backpropagation, compute gradients
    opti.step() # apply gradients

    pred=torch.max(out,1)[1]
    print('acc=',(pred==label).sum().item()/200*100,'%')