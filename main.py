import augmentation as augment
import torch
import torchvision.transforms as trans
import time
from dataset import ContourDataset


if torch.cuda.is_available():
    device = 'cuda'
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    #torch.set_default_tensor_type('torch.FloatTensor')


img_path = './ContourDrawing/image'
skt_path = './ContourDrawing/sketch-rendered/width-5'
list_path = './ContourDrawing/list/train.txt'

transformer = augment.Compose([
    augment.HFlip(0.5),
    augment.Resize((286,286)),
    augment.Rotation_and_Crop(0.8),
    augment.ToTensor(),
    augment.Crop(258),
    augment.Normalize()
])

custom_contour = ContourDataset(img_path,skt_path,list_path,transformer=transformer)

custom_loader = torch.utils.data.DataLoader(
    dataset=custom_contour,
    batch_size=2,
    shuffle=True)

dataset_size = len(custom_loader)
print(dataset_size)

epochs = 1000


for e in range(epochs):
    epoch_start_time = time.time()

    for i, (image, targets) in enumerate(custom_loader):



def test(image, targets):
    print(image.shape)
    print(targets[0].shape)
    image = trans.ToPILImage()(image[0])
    target = trans.ToPILImage()(targets[0][0])
    image.save('test.jpg')
    target.save('test_target.jpg')