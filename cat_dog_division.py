from PIL import Image
import torch
import torchvision.transforms as transforms

net = torch.load("net_9.pth",weights_only=False)
net.eval()

classes = [
    "airplane","automobile","bird",
    "cat","deer","dog","frog",
    "horse","ship","truck"
]

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

img_dir = "./cat_dog/cat"

correct=0
total=0

for i in range(22):
    img_path = f"{img_dir}/cat.{i}.jpg"
    img = Image.open(img_path)
    img = transform(img)
    img = torch.reshape(img,(1,3,32,32))

    with torch.no_grad():
        output = net(img)
        probability = torch.softmax(output,dim=1)
        top3_prob,top3_index = torch.topk(probability,3)

    print(f"预测图片cat.{i}的预测结果：")
    for j in range(3):
        class_name = classes[top3_index[0][j]]
        prob = top3_prob[0][j]

        print(
            f"  {class_name}: {prob:.2%}"
        )



#print("-"*17)
#print(f"预测{total}张猫的图片，正确识别了{correct}张，准确率：{accuracy:.2%}")
