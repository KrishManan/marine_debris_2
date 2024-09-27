import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import time

from resnet import resnet18


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "../Cropped_Data/benchmark_images/26.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    class_indict = ["1", "2", "3"]

    # create model
    model = resnet18(num_classes=3).to(device)

    # load model weights
    weights_path = "../Weights/Resnet18best.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        # predict class
        start=time.time()
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        print(predict)
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
