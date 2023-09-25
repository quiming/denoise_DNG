import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from torchviz import make_dot
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)            #通过1X1卷积核 使得in_channels 变为 c2[0]
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1)
        self.p3_3 = nn.Conv2d(c3[1], c3[1], kernel_size=3, padding=1)


        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_3(self.p3_2(F.relu(self.p3_1(x)))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)

class Unet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(Unet, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)


        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.Drop = nn.Dropout(0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)


        self.conv3 = Inception(64, 32, (32, 64), (8, 16), 16)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.Drop = nn.Dropout(0.5)
        self.pool3 = nn.MaxPool2d(kernel_size=2)


        self.conv4 = Inception(128, 64, (64, 128), (16, 32), 32)
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.Drop = nn.Dropout(0.5)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        n, c, h, w = x.shape
        #print('x:',x.shape)    #x: torch.Size([1, 4, 512, 512])
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_image = F.pad(x, (0, w_pad, 0, h_pad), 'replicate')
        #print('pad_i',padded_image.shape) #pad_i torch.Size([1, 4, 512, 512])
        conv1 = self.leaky_relu(self.conv1_1(padded_image))
        conv1 = self.leaky_relu(self.conv1_2(conv1))
        conv1 = self.Drop(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.leaky_relu(self.conv2_1(pool1))
        conv2 = self.leaky_relu(self.conv2_2(conv2))
        conv2 = self.Drop(conv2)
        pool2 = self.pool1(conv2)

        conv3 = self.leaky_relu(self.conv3(pool2))
        conv3 = self.leaky_relu(self.conv3_1(conv3))
        conv3 = self.Drop(conv3)
        pool3 = self.pool1(conv3)

        conv4 = self.leaky_relu(self.conv4(pool3))
        conv4 = self.leaky_relu(self.conv4_1(conv4))
        conv4 = self.Drop(conv4)

        up7 = self.upv7(conv4)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.leaky_relu(self.conv7_1(up7))
        conv7 = self.leaky_relu(self.conv7_2(conv7))
        conv7 = self.Drop(conv7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.leaky_relu(self.conv8_1(up8))
        conv8 = self.leaky_relu(self.conv8_2(conv8))
        conv8 = self.Drop(conv8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.leaky_relu(self.conv9_1(up9))
        conv9 = self.leaky_relu(self.conv9_2(conv9))
        conv9 = self.Drop(conv9)

        conv10 = self.conv10_1(conv9)
        out = conv10[:, :, :h, :w]
        return out

    def leaky_relu(self, x):
        out = torch.max(0.2 * x, x)
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0')
    test_input = torch.from_numpy(np.random.randn(1, 4, 217, 289)).float().to(device)
    net = Unet().to(device)
    output = net(test_input)
    # MyConvNetVis = make_dot(output, params=dict(list(net.named_parameters()) + [('test_input', test_input)]))
    # MyConvNetVis.format = 'png'
    # # 指定文件生成的文件夹
    # MyConvNetVis.directory = "data"
    # # 生成文件
    # MyConvNetVis.view()
    print(output.shape)  #torch.Size([1, 4, 1736, 2312])
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
    #官方给的网络参数:Total number of paramerters in networks is 7760484
    #              Total number of paramerters in networks is 10217716

