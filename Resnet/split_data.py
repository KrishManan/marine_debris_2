from ast import arg
from multiprocessing import parent_process
import os
from shutil import copy, rmtree
import random
import argparse

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main(args):
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到测试集中
    split_rate = args.ratio

    # 指向读取的文件夹
    origin_dataset = args.readpath
    assert os.path.exists(origin_dataset)
    fish_class = [cla for cla in os.listdir(origin_dataset)
                    if os.path.isdir(os.path.join(origin_dataset, cla))]

    # 建立保存训练集的文件夹
    up_one_path = '/'.join(origin_dataset.split('/')[:-1])
    train_root = os.path.join(up_one_path, "train")
    mk_file(train_root)
    for cla in fish_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    test_root = os.path.join(up_one_path, "test")
    mk_file(test_root)
    for cla in fish_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(test_root, cla))

    for cla in fish_class:
        cla_path = os.path.join(origin_dataset, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(test_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-readpath', type=str,
                        default='./dataset')
    parser.add_argument('-ratio', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
