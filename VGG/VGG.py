import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import os
import torch.backends.cudnn as cudnn

dic = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
        5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
start_time = time.time()
batch_size = 128
learning_rate = 0.1

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                          std=(0.2471, 0.2436, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                          std=(0.2471, 0.2436, 0.2616))
])


train_dataset = datasets.CIFAR10(root='/home/ljj0512/private/workspace/cifar-10-batches-py',
                                 train=True,
                                 transform=transform_train,
                                 download=True)

test_dataset = datasets.CIFAR10(root='/home/ljj0512/private/workspace/cifar-10-batches-py',
                                train=False,
                                transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)



class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential( 
            ## 9개의 conv, 1개의 fc = 10
            # 32 x 32

            nn.Conv2d(3, 64, kernel_size=3, padding=1), ###### 01 ######
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1), ###### 02 ######
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1), ###### 03 ######
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1), ###### 04 ######
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1), ###### 05 ######
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1), ###### 06 ######
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), ###### 07 ######
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4 x 4

            nn.Conv2d(256, 512, kernel_size=3, padding=1), ###### 08 ######
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), ###### 09 ######
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2 x 2
        )
        self.classifier = nn.Linear(2048, num_classes) # 512 * 2 * 2 = 2048

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x.size()=[batch_size, channel, width, height] 
        #          [128, 512, 2, 2] 
        # flatten 결과 => [128, 512x2x2]
        x = self.classifier(x)
        return x


model = VGG()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().cuda()

if torch.cuda.device_count() > 0:
    print("USE", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    cudnn.benchmark = True
else:
    print("USE ONLY CPU!")

if torch.cuda.is_available():
    model.cuda()

train_loss_graph = []
def train(epoch):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
        
        # 모든 gradient를 0으로 set해주는 것은 RNN의 경우를 대비하여 function 자체가 gradient를 accumulate하도록 만들어졌기 때문
        optimizer.zero_grad()
        # forward()한 값을 반환 
        output = model(data)
        loss = criterion(output, target)
        # When you call loss.backward(),
        # all it does is compute gradient of loss w.r.t all the parameters in loss
        # that have requires_grad = True and store them in parameter.grad attribute for every parameter.
        # optimizer.step() updates all the parameters based on parameter.grad
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        # torch.max() : (maximum value, index of maximum value) return. 
        # 1 :  row마다 max계산 (즉, row는 10개의 class를 의미) 
        # 0 : column마다 max 계산 
        total += target.size(0)
        # 가장 높은 값이 나온 클래스들과 target(label)을 비교하여 correct에 더함.
        correct += predicted.eq(target.data).cpu().sum() 
        if not(epoch==0 and batch_idx==0):
          train_loss_graph.append(train_loss/(batch_idx+1))
        if batch_idx % 10 == 0:        
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

test_loss_graph = []
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        outputs = model(data)
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
    test_loss_graph.append(test_loss/(batch_idx+1))
    print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
      .format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in range(0, 5): #165):
    # if epoch < 80:
    #     learning_rate = learning_rate
    # elif epoch < 120:
    #     learning_rate = learning_rate * 0.1
    # else:
    #     learning_rate = learning_rate * 0.01
    # for param_group in optimizer.param_groups:
    #     param_group['learning_rate'] = learning_rate

    train(epoch)
    test()
    print('\n')


import matplotlib.pyplot as plt

plt.plot(train_loss_graph)
plt.title('train loss')
plt.savefig('tarin_loss_graph.jpg')
plt.close()

plt.plot(test_loss_graph)
plt.title('test loss')
plt.savefig('test_loss_graph.jpg')
plt.close()

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))