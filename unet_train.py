import os
import time
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn

class Eyedataset(Dataset):
    def __init__(self, image_folder,label_folder,transform,label_transform,img_size=572):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.img_names=os.listdir(self.image_folder)
        self.img_size=img_size

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name=self.img_names[idx]
        img_path = os.path.join(self.image_folder, img_name)#拼接文件夹名和文件名
        image = Image.open(img_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        label_path=os.path.join(self.label_folder, label_name)
        with open(label_path,"r",encoding="utf-8") as f:
            label_content=f.read()
            object_infos=label_content.strip().split("\n")
            target=[]
            for object_info in object_infos:
                info_list=object_info.strip().split(" ")
                class_id=float(info_list[0])
                center_x=float(info_list[1])
                center_y=float(info_list[2])
                width=float(info_list[3])
                height=float(info_list[4])
                target.extend([class_id,center_x,center_y,width,height])
            target=torch.tensor(target)
            mask = torch.from_numpy(mask)  # [H, W]
        return image, mask


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0)  # 由572*572*1变成了570*570*64
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)  # 由570*570*64变成了568*568*64
        self.relu1_2 = nn.ReLU(inplace=True)

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 采用最大池化进行下采样，图片大小减半，通道数不变，由568*568*64变成284*284*64

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)  # 284*284*64->282*282*128
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)  # 282*282*128->280*280*128
        self.relu2_2 = nn.ReLU(inplace=True)

        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 采用最大池化进行下采样  280*280*128->140*140*128

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)  # 140*140*128->138*138*256
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)  # 138*138*256->136*136*256
        self.relu3_2 = nn.ReLU(inplace=True)

        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 采用最大池化进行下采样  136*136*256->68*68*256

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)  # 68*68*256->66*66*512
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)  # 66*66*512->64*64*512
        self.relu4_2 = nn.ReLU(inplace=True)

        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 采用最大池化进行下采样  64*64*512->32*32*512

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)  # 32*32*512->30*30*1024
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)  # 30*30*1024->28*28*1024
        self.relu5_2 = nn.ReLU(inplace=True)

        # 接下来实现上采样中的up-conv2*2
        self.up_conv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0) # 28*28*1024->56*56*512


        self.conv6_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0)  # 56*56*1024->54*54*512
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)  # 54*54*512->52*52*512
        self.relu6_2 = nn.ReLU(inplace=True)

        self.up_conv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0) # 52*52*512->104*104*256

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)  # 104*104*512->102*102*256
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)  # 102*102*256->100*100*256
        self.relu7_2 = nn.ReLU(inplace=True)

        self.up_conv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0) # 100*100*256->200*200*128


        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)  # 200*200*256->198*198*128
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)  # 198*198*128->196*196*128
        self.relu8_2 = nn.ReLU(inplace=True)

        self.up_conv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0) # 196*196*128->392*392*64


        self.conv9_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)  # 392*392*128->390*390*64
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)  # 390*390*64->388*388*64
        self.relu9_2 = nn.ReLU(inplace=True)

        # 最后的conv1*1
        self.conv_10 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)

    # 中心裁剪，
    def crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        # 如果原始张量的尺寸为10，而delta为2，那么"delta:tensor_size - delta"将截取从索引2到索引8的部分，长度为6，以使得截取后的张量尺寸变为6。
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.relu1_1(x1)
        x2 = self.conv1_2(x1)
        x2 = self.relu1_2(x2)  # 这个后续需要使用
        down1 = self.maxpool_1(x2)

        x3 = self.conv2_1(down1)
        x3 = self.relu2_1(x3)
        x4 = self.conv2_2(x3)
        x4 = self.relu2_2(x4)  # 这个后续需要使用
        down2 = self.maxpool_2(x4)

        x5 = self.conv3_1(down2)
        x5 = self.relu3_1(x5)
        x6 = self.conv3_2(x5)
        x6 = self.relu3_2(x6)  # 这个后续需要使用
        down3 = self.maxpool_3(x6)

        x7 = self.conv4_1(down3)
        x7 = self.relu4_1(x7)
        x8 = self.conv4_2(x7)
        x8 = self.relu4_2(x8)  # 这个后续需要使用
        down4 = self.maxpool_4(x8)

        x9 = self.conv5_1(down4)
        x9 = self.relu5_1(x9)
        x10 = self.conv5_2(x9)
        x10 = self.relu5_2(x10)

        # 第一次上采样，需要"Copy and crop"（复制并裁剪）
        up1 = self.up_conv_1(x10)  # 得到56*56*512
        # 需要对x8进行裁剪，从中心往外裁剪
        crop1 = self.crop_tensor(x8, up1)
        up_1 = torch.cat([crop1, up1], dim=1)

        y1 = self.conv6_1(up_1)
        y1 = self.relu6_1(y1)
        y2 = self.conv6_2(y1)
        y2 = self.relu6_2(y2)

        # 第二次上采样，需要"Copy and crop"（复制并裁剪）
        up2 = self.up_conv_2(y2)
        # 需要对x6进行裁剪，从中心往外裁剪
        crop2 = self.crop_tensor(x6, up2)
        up_2 = torch.cat([crop2, up2], dim=1)

        y3 = self.conv7_1(up_2)
        y3 = self.relu7_1(y3)
        y4 = self.conv7_2(y3)
        y4 = self.relu7_2(y4)

        # 第三次上采样，需要"Copy and crop"（复制并裁剪）
        up3 = self.up_conv_3(y4)
        # 需要对x4进行裁剪，从中心往外裁剪
        crop3 = self.crop_tensor(x4, up3)
        up_3 = torch.cat([crop3, up3], dim=1)

        y5 = self.conv8_1(up_3)
        y5 = self.relu8_1(y5)
        y6 = self.conv8_2(y5)
        y6 = self.relu8_2(y6)

        # 第四次上采样，需要"Copy and crop"（复制并裁剪）
        up4 = self.up_conv_4(y6)
        # 需要对x2进行裁剪，从中心往外裁剪
        crop4 = self.crop_tensor(x2, up4)
        up_4 = torch.cat([crop4, up4], dim=1)

        y7 = self.conv9_1(up_4)
        y7 = self.relu9_1(y7)
        y8 = self.conv9_2(y7)
        y8 = self.relu9_2(y8)

        # 最后的conv1*1
        out = self.conv_10(y8)
        return out

if __name__ == '__main__':
    train_data = Eyedataset(image_folder="D:\\yolo_pytorch\\Dataset\\train\\images",label_folder="D:\\yolo_pytorch\\Dataset\\train\\labels", transform = transforms.Compose([transforms.Resize((572, 572)),transforms.ToTensor()]),label_transform=transforms.ToTensor(),img_size=388)
    test_data = Eyedataset(image_folder="D:\\yolo_pytorch\\Dataset\\val\\images",label_folder="D:\\yolo_pytorch\\Dataset\\val\\labels", transform = transforms.Compose([transforms.Resize((572, 572)),transforms.ToTensor()]),label_transform=transforms.ToTensor(),img_size=388)

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    unet = Unet()
    unet=unet.cuda()

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()

    optim = torch.optim.SGD(unet.parameters(), lr=1e-3,momentum=0.9)

    train_step = 0
    test_step = 0
    epoch = 10

    unet.train()
    for i in range(epoch):
        print("-------第{}次训练-------".format(i + 1))
        start = time.time()
        for data in train_dataloader:
            imgs, target = data
            imgs = imgs.cuda()
            target = target.cuda()
            output = unet(imgs)
            loss = loss_fn(output, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_step += 1
            if train_step % 100 == 0:
                print("训练次数:{},训练loss：{}".format(train_step,loss.item()))
        end = time.time()
        print("耗时:{}".format(end - start))

        unet.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.cuda()
                targets = targets.cuda()
                output = unet(imgs)
                loss = loss_fn(output, targets)
                test_loss += loss.item()
        print("整体测试集的loss:{}".format(test_loss))
        torch.save(unet.state_dict(), 'unet_{}.pth'.format(i))
        print("模型已保存")












