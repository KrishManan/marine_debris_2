
from ntpath import join
import random
import os
import time
from requests import head
from torch.autograd import Variable
from PIL import Image

import torch
import shutil
import pandas as pd
import itertools
import csv
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def read_split_data(root: str, val_rate: float = 0.25):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    plankton_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    plankton_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(plankton_class))

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in plankton_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    return train_images_path, train_images_label, val_images_path, val_images_label

def read_csv_data(root: str, csv_file: str, val_rate: float = 0.25):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset path: {} does not exist.".format(root)
    assert os.path.exists(csv_file), "csv_file path: {} does not exist.".format(csv_file)

    # 读取csv文件
    labels = pd.read_csv(csv_file)
    labels['id'] = labels['id'].apply(lambda x: x + '.jpg')
    labels_sorted = labels.sort_values(by='breed', ascending=True)

    # 生成类别名称以及对应的数字索引
    class_indices = dict()
    for i, label in enumerate(labels_sorted.breed.unique()):
        class_indices[label] = i
    
    with open('./classes_indices.txt', 'w') as f:
        for key in class_indices.keys():
            f.write(key + '\n')
    
    # 按比例随机划分训练集和验证集
    val_images_index = random.sample(list(range(len(labels))), k=int(len(labels) * val_rate))
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    
    for index in range(len(labels)):
        if index in val_images_index:
            val_images_path.append(os.path.join(root, labels['id'][index]))
            val_images_label.append(class_indices[labels['breed'][index]])
        else:
            train_images_path.append(os.path.join(root, labels['id'][index]))
            train_images_label.append(class_indices[labels['breed'][index]])
    
    return train_images_path, train_images_label, val_images_path, val_images_label

def train(train_loader, batch_size, net, device, optimizer, loss_function, train_scheduler, writer, epoch, tb=True):
    
    start = time.time()
    net.train()
    
    train_loss = 0.0  # cost function error
    correct = 0.0
    
    for batch_index, (images, labels) in enumerate(train_loader):
        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        loss.backward()
        optimizer.step()

        print('\rTraining Epoch: {epoch} [{trained_samples}/{total_samples}]\t\tLoss: {:10.6f}\t\tLR: {:10.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * batch_size + len(images),
            total_samples=len(train_loader.dataset)), end='')

    # 更新学习率，注意放置的位置
    train_scheduler.step()

    finish = time.time()

    print('\nepoch {} training time consumed: {:.3f}s'.format(epoch, finish - start))
    
    if tb:
        writer.add_scalar('Train/Loss', train_loss / len(train_loader.dataset), epoch)
        writer.add_scalar('Train/Accuracy', correct.float() / len(train_loader.dataset), epoch)

def eval_training(val_loader, net, device, loss_function, writer, epoch, tb=True):
    
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    # 显示GPU占用情况
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(val_loader.dataset),
        correct.float() / len(val_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Val/Loss', test_loss / len(val_loader.dataset), epoch)
        writer.add_scalar('Val/Accuracy', correct.float() / len(val_loader.dataset), epoch)

    return correct.float() / len(val_loader.dataset)

def test_model(test_loader, classes_name, net, device):
    
    net.eval()

    # 展示结果的准确率
    correct_1 = 0.0
    correct_5 = 0.0
    
    class_correct = [0 for i in range(len(classes_name))]
    class_total = [0 for i in range(len(classes_name))]
    class_acc = {}
        
    for n_iter, (images, labels) in enumerate(test_loader):
        print("\riteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)), end='')

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        
        _, predicted = torch.max(outputs, dim=1)
        c = (predicted == labels)
        
        # 各个类别的准确率
        for j in range(len(labels)):
            l = labels[j]
            class_correct[l] += c[j].item()
            class_total[l] += 1
        
        # 计算top1和top5准确率
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)
        
        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()
        
        #compute top1
        correct_1 += correct[:, :1].sum()

        #compute top 5
        correct_5 += correct[:, :5].sum()
    
    top1_acc = str(round((100 * correct_1 / len(test_loader.dataset)).item(), 3)) + '%'
    top5_acc = str(round((100 * correct_5 / len(test_loader.dataset)).item(), 3)) + '%'
    
    for i in range(len(classes_name)):
        class_acc[classes_name[i]] = str(round(100 * class_correct[i] / class_total[i], 3)) + '%'
    
    return top1_acc, top5_acc, class_acc
    
def infer(dataset_root, infer_images_path, data_transforms, classes_name, net, device, savepath):
    
    net.eval()
    
    # 打开submission.csv文件并写入
    f = open(os.path.join(savepath, 'submission.csv'), 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    
    # 写入表头
    header = ['id'] + classes_name
    writer.writerow(header)
    
    for i in tqdm(range(len(infer_images_path))):
        
        # 写入文件的data
        data = []
        
        # 图像读取路径 
        image_path = infer_images_path[i]
        abs_image_path = os.path.join(dataset_root, image_path)
        
        # 写入图片文件名称
        data.append(os.path.splitext(image_path)[0])
        
        # 图像预处理
        image = Image.open(abs_image_path).convert('RGB')
        image_tensor = data_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        
        # 模型推理
        input = Variable(image_tensor).to(device)
        output = net(input)
        res = list(torch.softmax(output, dim=1).cpu().numpy()[0])
        res = ['{:.17f}'.format(num) for num in res]
                
        # 结果存储
        data.extend(res)
        writer.writerow(data)

        index = res.index(max(res))
        save_class_path = os.path.join(savepath, classes_name[index])
        if os.path.exists(save_class_path):
            shutil.copy(abs_image_path, save_class_path)
        else:
            os.mkdir(save_class_path)
            shutil.copy(abs_image_path, save_class_path)
        
    f.close()
        

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, model_name, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds,
                          text_fontsize=6,
                          tick_fontsize=4):
    """
    This function prints and plots the confusion matrix.
    cm is confusion matrix
    Normalization can be applied by setting `normalize=True`.
    we can choose different colors cmap=plt.cm.Blues
    """
    title = title+': '+model_name
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45,fontsize=tick_fontsize)
    plt.xticks(tick_marks, classes, fontsize=tick_fontsize)
    plt.yticks(tick_marks, classes, fontsize=tick_fontsize)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=text_fontsize)
        ######这⾥里里需要注意不不能让text的fontsize太⼤大，否则数字将重叠######
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')