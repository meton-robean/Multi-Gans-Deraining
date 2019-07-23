
import numpy as np
import os
from PIL import Image
import time
import argparse
from data import get_dataloader, read_data, read_data_test2
import metrics
import torch
from models import LPAGAN, vgg_init
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from tensorboardX import SummaryWriter
#import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Self-Attention Derain GAN trainer')
parser.add_argument('--batch', default=32, type=int, help='batch size')
parser.add_argument('--start_epoch', default=0, type=int, help='maximum iterations')
parser.add_argument('--end_epoch', default=200000, type=int, help='maximum iterations')
parser.add_argument('--code', default=128, type=int, help='size of code to input generator')
parser.add_argument('--lr_g', default=1e-3, type=float, help='learning rate of generator')
parser.add_argument('--lr_d', default=4e-3, type=float, help='learning rate of discriminator')
parser.add_argument('--n_d',  default=1,  type=int,  help=('number of discriminator update ' 'per 1 generator update'))
parser.add_argument('--n_level', default=3, type=int, help='levels of LPAGAN')
parser.add_argument('--model', default='dcgan', choices=['dcgan', 'resnet'], help='choice model class')
parser.add_argument('--rain_path', default='/home/cmt/Baiduyun FIles/All_My_Files/All_My_DataSets/derain_datasets/rainy_image_dataset/training/rainy_image', type=str, help='Path to rain image directory')
parser.add_argument('--gt_path', default='/home/cmt/Baiduyun FIles/All_My_Files/All_My_DataSets/derain_datasets/rainy_image_dataset/training/ground_truth', type=str, help='Path to gt image directory')
parser.add_argument('--GamaA', default= 1e-3, help='weight of loss G')
parser.add_argument('--GamaP', default= 5e-3, help = 'weight of loss P')
parser.add_argument('--checkpoint_path', default='/home/cmt/Baiduyun FIles/All_My_Files/All_My_Coding_Tests/ML_class_derain_proj/Derain_ML_proj_v2.8/checkpoint', type=str, help='path to latest checkpoint (default: none)')  #导入保存的checkpoint


#print('is cuda available:', torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform4train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


###########################################################################################
def Adversary_loss(logits_real, logits_fake):

    # Batch size.
    N = logits_real.size()
    true_labels = torch.Tensor(torch.zeros(N)).to(device)  # A(Ico)为0，A（Ien）为1
    MSE_loss = nn.MSELoss()
    Ico_loss = MSE_loss(logits_real, true_labels)  # 识别正确的为正确
    Ien_loss = MSE_loss(logits_fake, 1 - true_labels)  # 识别错误的为错误
    loss = Ico_loss + Ien_loss
    return loss

def Generator_loss(logits_fake):
    # Batch size.
    N = logits_fake.size()
    true_labels = torch.Tensor(torch.zeros(N)).to(device)  # A（Ien）为0
    # 计算生成器损失
    MSE_loss = nn.MSELoss()
    loss = MSE_loss(logits_fake, true_labels)
    return loss

def Vgg_loss(vgg, fake_residual, gt_residual):
    fake_features = vgg(fake_residual)
    gt_features = vgg(gt_residual)
    mseloss = nn.MSELoss()
    for i in range(len(fake_features)):
        fake_f = Variable(fake_features[i].data, requires_grad = False).to(device)
        gt_f = Variable(gt_features[i].data, requires_grad = False).to(device)
        if(i==0):
            loss = mseloss(fake_f, gt_f)
        else:
            loss = loss + mseloss(fake_f, gt_f)
    return loss

################################################################################



def train(args, LPAGAN):
    #dataloader = get_dataloader(gt_path=args.gt_path, rain_path=args.rain_path, batchSize=args.batch)

    optim_D = []
    optim_G = []
    z =[]
    for i in range(LPAGAN.n_level):

        optim_G.append(optim.Adam(filter(lambda p: p.requires_grad, LPAGAN.Generator[i].parameters()), lr = args.lr_g))
        optim_D.append(optim.Adam(filter(lambda p: p.requires_grad, LPAGAN.Discriminator[i].parameters()), lr = args.lr_d))
        if(i ==0):
            z.append(0.1)
        else:
            z.append(z[i-1]*2)

    mseloss = nn.MSELoss()
    Vgg16 = vgg_init.vgg(vgg_init.vgg_in())
    counter = 0
    #log writer
    writer = SummaryWriter('./logs')

    # resume from checkpoint

    if args.checkpoint_path:
        print('load checkpoint from %s ....'%args.checkpoint_path)
        train_state_checkpoint = torch.load(args.checkpoint_path+'/train_state.pkl')
        args.start_epoch = train_state_checkpoint['epoch']
        for idx in range(LPAGAN.n_level):   #前四个导入预训练模型
            gen_checkpoint = torch.load(args.checkpoint_path+'/LPAGAN_Generator_{}.pkl'.format(i))
            print('load '+args.checkpoint_path+'/LPAGAN_Generator_{}.pkl'.format(i))
            dis_checkpoint = torch.load(args.checkpoint_path+'/LPAGAN_Discriminator_{}.pkl'.format(i))
            print('load ' + args.checkpoint_path + '/LPAGAN_Discriminator_{}.pkl'.format(i))

            LPAGAN.Generator[idx].load_state_dict(gen_checkpoint)
            LPAGAN.Discriminator[idx].load_state_dict(dis_checkpoint)
        print('load checkpoint file done!')

    '''
    validation_data, validation_label = read_data_test2(input_path=args.rain_path, gt_path=args.gt_path, size_input=64, num_channel=3,
                                                   batch_size=3)  # data for validation
    print("check patch pair:")
    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.imshow(validation_data[i, :, :, :])
        plt.title('input')
        plt.subplot(3, 2, 2*i+2)
        plt.imshow(validation_label[i, :, :, :])
        plt.title('ground truth')
    plt.show()
    '''
    for i in range(args.start_epoch, args.end_epoch):
        #for k, data in enumerate(dataloader):
        #使用更加灵活的read_data，替换dataloader
        data = read_data(gt_path=args.gt_path, rain_path=args.rain_path, num_channel=3, batch_size=args.batch, size_input=64, transform=transform4train)
        rain_image, gt_image = data
        rain_image = rain_image.to(device)  # change to cuda data
        gt_image = gt_image.to(device)

        last_rain_image = rain_image

        start_time = time.time()

        loss1 = []
        loss2 = []

        for j in range(LPAGAN.n_level):
            gen_image = LPAGAN.Generator[j](last_rain_image, z[j])
            logits_fake = LPAGAN.Discriminator[j](gen_image)
            logits_real = LPAGAN.Discriminator[j](gt_image)

            lossd = Adversary_loss(logits_real, logits_fake)
            optim_D[j].zero_grad()
            lossd.backward()
            optim_D[j].step()
            loss1.append(lossd)



            gen_image = LPAGAN.Generator[j](last_rain_image, z[j])
            logits_fake = LPAGAN.Discriminator[j](gen_image)

            img_mseloss = mseloss(gen_image, gt_image)
            gen_loss = Generator_loss(logits_fake)
            pertul_loss = Vgg_loss(Vgg16, gen_image, gt_image)

            #print('img_mseloss: ', img_mseloss, 'gen_loss: ', gen_loss, 'pertul_loss: ', pertul_loss )
            lossg = img_mseloss+  args.GamaA *gen_loss + z[j]*args.GamaP * pertul_loss

            optim_G[j].zero_grad()
            lossg.backward()
            optim_G[j].step()
            loss2.append(lossg)

            last_rain_image = gen_image.detach() #

            counter = counter + 1

            if (counter) % (LPAGAN.n_level*5) == 0:
                print("Iter: [%2d], step: [%2d], time: [%4.4f]" \
                      % ((i + 1), counter, time.time() - start_time))
                print("Generator loss is ", loss2)
                print("Discriminator loss is ", loss1)
                print("*********************************")

            if (counter) % (LPAGAN.n_level*100) == 0:
                #save state of training process
                torch.save({
                        'epoch': i+1
                     },
                    args.checkpoint_path+'/train_state.pkl')
                #save model weight
                for n in range(LPAGAN.n_level):
                    torch.save(LPAGAN.Generator[n].state_dict(), args.checkpoint_path+'/LPAGAN_Generator_' + str(n) + '.pkl')
                    torch.save(LPAGAN.Discriminator[n].state_dict(), args.checkpoint_path+'/LPAGAN_Discriminator_' + str(n) + '.pkl')

        if i%50 == 0:  #save log
            for n in range(LPAGAN.n_level):
                writer.add_scalars('data/g_d_loss_group{}'.format(n), {'gen{}_loss'.format(n): loss2[n],
                                                          'disc{}_loss'.format(n): loss1[n]
                                                     }, i )
                #log gradient
                '''
                for name, param in LPAGAN.Generator[n].named_parameters():
                    #writer.add_histogram('gen{}_'.format(n)+name + '_grad', param.clone().grad.cpu().data.numpy(), i)
                    writer.add_histogram('gen{}_'.format(n)+name + '_data', param.clone().cpu().detach().numpy(), i)

                for name2, param2 in LPAGAN.Discriminator[n].named_parameters():
                    #writer.add_histogram('disc{}_'.format(n)+name + '_grad', param.clone().grad.cpu().data.numpy(), i)
                    writer.add_histogram('disc{}_'.format(n)+name2 + '_data', param2.clone().cpu().detach().numpy(), i)
                '''



if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    lpagan = LPAGAN(args.n_level)
    train(args, lpagan)



