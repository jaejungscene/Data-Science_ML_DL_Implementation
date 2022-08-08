import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        tmp = x
        x = x + 1
        return x, tmp


input = torch.randn([1000,1000]).cuda()
print(input)
model = MyModel()
model = nn.DataParallel(model, device_ids=[5,6,7])
output, tmp = model(input)
print('output:',output)
print('tmp:',tmp)