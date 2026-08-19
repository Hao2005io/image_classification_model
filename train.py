import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10(root="./data",train=True,
                                         transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data",train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                          download=True)
print(f"训练数据集长度：{len(train_data)}")
print(f"测试数据集长度：{len(test_data)}")

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

writer = SummaryWriter("./logs_train")
net = Net()
#损失函数
loss_fn = nn.CrossEntropyLoss()
#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10
for i in range(epoch):
    print(f"------第{i}轮训练开始------")
    for data in train_dataloader:
        imgs, targets = data
        output = net(imgs)
        loss = loss_fn(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数:{total_train_step},Loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            output = net(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy=(output.argmax(1)==targets).sum()
            total_accuracy += accuracy.item()
    print(f"整体测试集上的Loss:{total_test_loss}")
    print(f"整体测试集的正确率{total_accuracy/len(test_data)}")
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/len(test_data),total_test_step)
    total_test_step+=1
    torch.save(net,f"net_{i}.pth")
    print("模型已保存")
writer.close()