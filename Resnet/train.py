import argparse
from re import L

from utils import read_split_data, train, eval_training
import math
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from my_dataset import MyDataSet
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

# from models.lenet import LeNet
from resnet import resnet50
# from models.alexnet import AlexNet
# from resnet import resnet50
# from models.vggnet import vgg16_bn

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str,
                        default='../Cropped_Data/train',
                        help='read path for training')
    parser.add_argument('-savepath', type=str,
                        default='./results')
    parser.add_argument('-net', type=str, required=False, default="resnet50")
    parser.add_argument('-cuda', type=str, default='cuda:0')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('-num_classes', type=int, default=3, help='num of classes')
    parser.add_argument('-epochs', type=int, default=200, help='epochs for training')
    args = parser.parse_args()
    
    print(args)
    
    # 选择设备-gpu or cpu
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    print(device)
    
    """数据集准备"""
    # 数据集路径
    dataset_root = args.dataset
    
    # 划分训练和验证集，返回图片路径列表
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(dataset_root)
    # 数据预处理
    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),  # 随机裁剪一个area然后再resize
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    # 制作数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transforms["train"])

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transforms['val'])
    
    # 加载数据集
    batch_size = args.batch_size
    num_workers = 8
    
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=val_data_set.collate_fn)
    
    """网络"""
    # 实例化网络
    # net = resnet18()
    # net = resnet50()
    net = resnet50(num_classes=args.num_classes)
    
    net.to(device)
    
    """定义损失函数和优化器"""
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # learning rate decay
    # x表示当前步数，在0-args.epochs之间变化
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    train_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    
    """结果记录"""
    # tensorboard
    TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    runs_dir = os.path.join('./results', args.net, TIME_NOW, 'runs')
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    writer = SummaryWriter(log_dir=runs_dir)
    
    # 权重保存
    checkpoint_dir = os.path.join('./results', args.net, TIME_NOW, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    best_acc = 0.0
    best_epoch = 0
    
    start = time.time()
    
    for epoch in range(1, args.epochs+1):
        
        # 模型训练
        train(train_loader=train_loader, batch_size=args.batch_size, net=net, device=device,
              optimizer=optimizer, loss_function=loss_function, train_scheduler=train_scheduler,
              writer=writer, epoch=epoch)
        
        # 模型验证
        with torch.no_grad():
            acc = eval_training(val_loader, net, device, loss_function, writer, epoch=epoch)
        
        if epoch > 10 and best_acc < acc:
            best_epoch = epoch
            best_acc = acc
            
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, args.net + '-best.pth'))
            
            continue
    
    end = time.time()
    
    print('best epoch: %d' % best_epoch)
    print('best val acc: %.5f' % best_acc)
    print('training total time: %.5f s.' % (end-start))
    writer.close()
        
