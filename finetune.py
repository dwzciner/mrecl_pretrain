import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import configs.classification.class_parser_finetune as class_parser_finetune
import utils
import model.learner as Learner

class CNN(nn.Module):
    def __init__(self, in_channels, out_dimension,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,channels,3,
                               2,0)
        self.conv2 = nn.Conv2d(channels,channels,3,
                               1,0)
        self.conv3 = nn.Conv2d(channels,channels,3,
                               2,0)
        self.conv4 = nn.Conv2d(channels,channels,3,
                               1,0)
        self.conv5 = nn.Conv2d(channels,channels,2,
                               1,0)
        self.conv_relu_stack = nn.Sequential(
            self.conv1,nn.ReLU(),
            self.conv2,nn.ReLU(),
            self.conv3,nn.ReLU(),
            self.conv4,nn.ReLU(),
            self.conv5,nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3 * channels,out_dimension)
        )
    def forward(self,x):
        logits = self.conv_relu_stack(x)
        return logits

def load_model(args):
    if args['model_path'] is not None:
        net = torch.load(args['model_path'],
                         map_location="cpu")
    else:
        net = CNN(3,64,256)
    return net

def train_iterator(img_list,y_list, device, model, opt):
    for img,y in zip(img_list,y_list):

        img = img.to(device)
        y = y.to(device)

        pred = model(img)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(pred[0], y.long())

        loss.backward()
        opt.step()

def eval_iterator(iterator, device, model):
    correct = 0
    for img, target in iterator:
        img = img.repeat(1,3,1,1)
        img = nn.functional.interpolate(img, size=(32, 32), mode='bilinear', align_corners=False)  # 调整大小
        img = img.to(device)
        target = target.to(device)
        logits_q = model(img)

        pred_q = (logits_q[0]).argmax(dim=1)

        correct += torch.eq(pred_q, target).sum().item() / len(img)
    return correct / len(iterator)

def filter(dataloader):
    img_list = []
    y_list = []
    count = {}

    for img, data in dataloader:
        label = data.item()
        count.setdefault(label,0)
        if count[label] <= 1:
            img = img.repeat(1,3,1,1)
            img = nn.functional.interpolate(img, size=(32, 32), mode='bilinear', align_corners=False)  # 调整大小
            img_list.append(img)
            y_list.append(data)
            count[label] += 1

    return img_list,y_list

def main(args):

    mnist_train = datasets.MNIST(root='../data/mnist',train=True,transform=transforms.ToTensor(),download=True)
    mnist_test = datasets.MNIST(root='../data/mnist',train=False,transform=transforms.ToTensor(),download=True)
    # 1 1 28 28
    train_loader = DataLoader(mnist_train,
                              batch_size=1,shuffle=True)
    test_loader = DataLoader(mnist_test,
                             batch_size=256,shuffle=True)

    img,y = filter(train_loader)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args)
    opt = torch.optim.Adam(model.parameters(),lr=0.001)
    model = model.to(device)

    for i in range(10):
        train_iterator(img, y,device,model,opt)

    correct = eval_iterator(test_loader,device,model)
    print(f'test acc : {correct}')

if __name__ == '__main__':
    p = class_parser_finetune.Parser()

    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)
    rank = p.parse_known_args()[0].rank
    args = utils.get_run(vars(p.parse_known_args()[0]), rank)
    main(args)

