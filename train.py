import os
import numpy as np
from net import ZSSRNet
from data import DataSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import PIL
import sys
from torchvision import transforms
import tqdm
import argparse

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train(model, img, target, num_batches, learning_rate, crop_size):
    loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    sampler = DataSampler(img, target, crop_size)
    model.cuda()
    with tqdm.tqdm(total=num_batches, miniters=1, mininterval=0) as progress:
        for iter, (hr, lr) in enumerate(sampler.generate_data()):
            model.zero_grad()

            lr = Variable(lr).cuda()
            hr = Variable(hr).cuda()

            output = model(lr) + lr
            error = loss(output, hr)

            cpu_loss = error.data.cpu().numpy().item()

            progress.set_description("Iteration: {iter} Loss: {loss}, Learning Rate: {lr}".format( \
                iter=iter, loss=cpu_loss, lr=learning_rate))
            progress.update()

            if iter > 0 and iter % 10000 == 0:
                learning_rate = learning_rate / 10
                adjust_learning_rate(optimizer, new_lr=learning_rate)
                print("Learning rate reduced to {lr}".format(lr=learning_rate) )

            error.backward()
            optimizer.step()

            if iter > num_batches:
                print('Done training.')
                break
            

def test(model, img, img_path):
    model.eval()
    
    save_name = os.path.splitext(os.path.basename(img_path))[0]

    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    input = Variable(img.cuda())
    residual = model(input)
    output = input + residual

    output = output.cpu().data[0, :, :, :]
    o = output.numpy()
    o[np.where(o < 0)] = 0.0
    o[np.where(o > 1)] = 1.0
    output = torch.from_numpy(o)
    output = transforms.ToPILImage()(output) 
    output.save(f'results/{save_name}_zssr.png')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_batches', type=int, default=15000, \
        help='Number of batches to run')
    parser.add_argument('--crop', type=int, default=128, \
        help='Random crop size')
    parser.add_argument('--lr', type=float, default=0.00001, \
        help='Base learning rate for Adam')
    parser.add_argument('--img', type=str, help='Path to input img')
    parser.add_argument('--target', type=str, help='Path to target img')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    img = PIL.Image.open(args.img)
    num_channels = len(np.array(img).shape)
    if num_channels == 3:
        model = ZSSRNet(input_channels = 3)
    elif num_channels == 2:
        model = ZSSRNet(input_channels = 1)
    else:
        print("Expecting RGB or gray image, instead got", img.size)
        sys.exit(1)

    target = PIL.Image.open(args.target)
    num_channels = len(np.array(target).shape)
    if num_channels == 3:
        model = ZSSRNet(input_channels = 3)
    elif num_channels == 2:
        model = ZSSRNet(input_channels = 1)
    else:
        print("Expecting RGB or gray image, instead got", target.size)
        sys.exit(1)

    # Weight initialization
    model.apply(weights_init_kaiming)

    train(model, img, target, args.factor, args.num_batches, args.lr, args.crop)
    test(model, target, args.factor, args.target)