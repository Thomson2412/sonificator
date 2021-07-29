import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image


class SceneDetectionVisual:

    def __init__(self, path, arch):
        self.arch = arch

        # load the pre-trained weights
        model_file = os.path.abspath(os.path.join(path, f"{arch}_places365.pth.tar"))

        self.model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # load the image transformer
        self.centre_crop = trn.Compose([
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load the class label
        categories_file_name = os.path.join(path, "categories_places365.txt")
        self.classes = list()
        with open(categories_file_name) as class_file:
            for line in class_file:
                self.classes.append(line.strip().split(' ')[0][3:])
        self.classes = tuple(self.classes)

    def detect(self, file_path):
        img = Image.open(file_path)
        crop = self.centre_crop(img)
        us = crop.unsqueeze(0)
        input_img = V(us)

        # forward pass
        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        _, idx = h_x.sort(0, True)
        return self.classes[idx[0]]

    def detect_bulk(self, input_dir):
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if ".jpg" in filename or ".png" in filename:
                    input_file_path = os.path.join(root, filename)
                    result = self.detect(input_file_path)
                    print(f"{filename}: {result}")
