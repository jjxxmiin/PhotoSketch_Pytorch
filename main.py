import augmentation as augment
from PIL import Image
import numpy as np
import os
from dataset import ContourDataset
from model import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# path
img_path = './ContourDrawing/image'
skt_path = './ContourDrawing/sketch-rendered/width-5'
list_path = './ContourDrawing/list/train.txt'

# require folder
if 'gen' not in os.listdir('.'):
    os.mkdir('./gen')
elif 'checkpoint' not in os.listdir('.'):
    os.mkdir('./checkpoint')

# augmentation
transformer = augment.Compose([
    augment.HFlip(0.5),
    augment.Resize((286, 286)),
    augment.Rotation_and_Crop(0.8),
    augment.ToTensor(),
    augment.Crop(256),
    augment.Normalize()
])

# dataset
custom_contour = ContourDataset(img_path, skt_path, list_path, transformer=transformer)

custom_loader = torch.utils.data.DataLoader(
    dataset=custom_contour,
    batch_size=1,
    shuffle=True)

# hyperparameter
epochs = 1000
lr = 0.0002
beta1 = 0.5
lambda_G = 1
lambda_A = 200

# model
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# optimizer
optim_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optim_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# loss
loss_GAN = GANLoss()
loss_L1 = torch.nn.L1Loss().to(device)

dataset_size = len(custom_loader)


def train_gen(real, targets, fake):
    """
    generator

    loss : min aggregate
    """
    optim_G.zero_grad()
    fake_cat = torch.cat((real, fake), 1)
    pred_fake = discriminator(fake_cat)
    loss_g_gan = loss_GAN(pred_fake, True) * lambda_G

    n = targets.shape[1]
    fake_expand = fake.expand(-1, n, -1, -1)
    l1 = torch.abs(fake_expand - targets)
    l1 = l1.view(-1, n, targets.shape[2] * targets.shape[3])
    l1 = torch.mean(l1, 2)
    min_l1, min_idx = torch.min(l1, 1)
    loss_g_l1 = torch.mean(min_l1) * lambda_A

    loss_g = loss_g_gan + loss_g_l1
    loss_g.backward()
    optim_G.step()

    return loss_g


def train_dis(real, targets, fake):
    """
    discriminator

    loss : mean aggregate
    """
    optim_D.zero_grad()
    fake_cat = torch.cat((real, fake), 1).detach()
    pred_fake = discriminator(fake_cat)
    loss_d_fake = loss_GAN(pred_fake, False)

    n = real.shape[1]
    loss_d_real_set = torch.empty(n, device=device)

    for i in range(n):
        sel = targets[:, i, :, :].unsqueeze(1)
        real_cat = torch.cat((real, sel), 1)
        pred_real = discriminator(real_cat)
        loss_d_real_set[i] = loss_GAN(pred_real, True)

    loss_d_real = torch.mean(loss_d_real_set)
    loss_d = (loss_d_fake + loss_d_real) * 0.5 * lambda_G
    loss_d.backward()
    optim_D.step()

    return loss_d


def save_model(net, path):
    torch.save(net.state_dict(), path)


def save_image(generation, path):
    img = generation.detach()[0][0].cpu().float().numpy()
    img = (img + 1) / 2.0 * 255.0
    image_pil = Image.fromarray(img.astype(np.uint8))
    image_pil.save(path)


# train
for e in range(epochs):
    for b, (image, targets) in enumerate(custom_loader):
        optim_G.zero_grad()

        real_A = image.to(device)
        fake_B = generator(real_A)
        real_B = targets.to(device)

        # discriminator
        loss_D = train_dis(real_A, real_B, fake_B)

        # generator
        loss_G = train_gen(real_A, real_B, fake_B)

        print("[Epoch %d/%d] [Batch %d/%d] [D loss %f] [G loss %f]" % (
        epochs, e, dataset_size, b, loss_D.item(), loss_G.item()))

        if b % 200 == 0:
            gen_img_path = "gen/gene_{}_{}.png".format(e, b)
            save_image(fake_B, gen_img_path)

    if e != 0 and e % 100 == 0:
        gen_model_path = "checkpoint/generator_{}.pth".format(e)
        dis_model_path = "checkpoint/discriminator_{}.pth".format(e)
        save_model(generator, gen_model_path)
        save_model(discriminator, dis_model_path)
