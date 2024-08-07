import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import time

from resnet import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./dataset/flower_dataset/val/roses/921984328_a60076f070_m.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    class_indict = ["0-daisy", "1-dandelion", "2-roses", "3-sunflowers", "4-tulips"]

    # create model
    model = resnet50(num_classes=3).to(device)

    # load model weights
    weights_path = "./results/AlexNet/Tuesday_28_May_2024_18h_57m_17s/checkpoints/AlexNet-best.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        start=time.time()
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        end=time.time()
        timedif=end-start
        print(timedif)

    print_res = "class: {}   prob: {:.3}".format(class_indict[predict_cla],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    plt.show()
    print(print_res)

if __name__ == '__main__':
    main()
