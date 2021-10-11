import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
import time

import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image as Image

CLASSES_TO_SHOW = 10

# create some regular pytorch model...
timest = time.time()
model = alexnet(pretrained=True).eval().cuda()
print("load time {}".format(time.time()-timest))

transforms = transforms.Compose([
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

path = './image.png'
raw_image = Image.open(path)
tensor_image = transforms(raw_image)
batch = torch.unsqueeze(tensor_image, 0)

timest = time.time()
output = model(batch)
print("recognition time{}".format(time.time()-timest))

sorted, indices = torch.sort(output, descending=True)
probability = F.softmax(output, dim=1)[0] * 100.0

with open('imagenet1000_clsidx_to_labels.txt') as labels:
    class_names = [i.strip() for i in labels.readlines()]

most_probable = [(class_names[i], probability[i].item()) for i in indices[0][:CLASSES_TO_SHOW]]
print("\nMost probable classes for this image:")
for i in range(CLASSES_TO_SHOW):
    print('{}: {:.4f}%'.format(most_probable[i][0], most_probable[i][1]))

# create example data
# x1 = torch.ones((1, 3, 224, 224)).cuda()
# x2 = torch.ones((1, 3, 224, 224)).cuda()
# x3 = torch.ones((1, 3, 224, 224)).cuda()

# timest = time.time()
# y1 = model(x1)
# print(time.time()-timest)

# timest = time.time()
# y2 = model(x2)
# print(time.time()-timest)

# timest = time.time()
# y3 = model(x3)
# print(time.time()-timest)
# print("y {}".format(y))





