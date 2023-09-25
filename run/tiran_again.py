import rawpy
import numpy as np
import torch
import time
from torch import nn,optim
#from network import Unet
from network import Unet
def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data

#反过来去训练得出噪声
def train(model,optimizer,loss_function,epochs,device1,device2):
    best_loss, best_epoch = 1, 0  # 输出验证集中准确率最高的轮次和准确率
    best_t_loss, best_t_epoch = 1, 0
    loss_list = []
    for epoch in range(epochs):
        print('==================第{}轮=================='.format(epoch + 1))
        s_time = time.time()
        loss_all = 0
        for i in range(90):
            #训练得到噪声
            raw_data = rawpy.imread('dataset/ground_truth/{}_gt.dng'.format(i)).raw_image_visible
            raw_label =rawpy.imread('dataset/noisy/{}_noise.dng'.format(i)).raw_image_visible
                #训练得到噪声
                #noise - gt     noise - noise +gt =gt
            h = 3472
            w = 4624
            black_level, white_level=1024,16383
            data_nor = normalization(raw_label,black_level,white_level)
            #print(data_nor.shape)
            image = torch.from_numpy(np.transpose(data_nor.reshape(-1, h//2, w//2, 4), (0, 3, 1, 2))).float()
            #print(image.shape)

            label_norm = normalization(raw_data ,h,w)  #标准化
                #label_norm = data_nor - label_norm
            label = torch.from_numpy(np.transpose(label_norm.reshape(-1, h//2, w//2, 4), (0, 3, 1, 2))).float()   #变成网络所需要的格式
                # image :torch.Size([1, 4, 868, 1156]) <class 'torch.Tensor'>
            x,y = image.to(device1),label.to(device1)
            model.to(device1)
            logits = model(x)  # 数据放入网络中
            loss = loss_function(logits, y)  # 得到损失值
            loss_all += loss.item() * x.size(0)
            optimizer.zero_grad()  # 优化器先清零，不然会叠加上次的数值
            loss.backward()  # 后向传播
            optimizer.step()

        loss_av = loss_all /90
        print(f'第{epoch + 1}轮，train损失值为：{loss_av}')
        # if epoch == 1:
        #     best_t_loss = loss_av
        if (epoch + 1) % 1 == 0:
            if best_t_loss > loss_av:
                best_t_loss = loss_av
                best_t_epoch = epoch + 1
                torch.save(model.state_dict(), 'model_train_100_20.pth')

        if (epoch+1) % 2 ==0:
            loss_val_all = 0
            model.eval()
            for i in range(90,100):
                    raw_data = rawpy.imread('dataset/ground_truth/{}_gt.dng'.format(i)).raw_image_visible
                    raw_label = rawpy.imread('dataset/noisy/{}_noise.dng'.format(i)).raw_image_visible
                    h = 1024
                    w = 16383
                    data_nor = normalization(raw_label, h, w)
                    image = torch.from_numpy(
                        np.transpose(data_nor.reshape(-1, h // 2, w // 2, 4), (0, 3, 1, 2))).float()

                    label_norm = normalization(raw_data, h, w)  # 标准化
                        #label_norm = data_nor - label_norm
                    label = torch.from_numpy(
                            np.transpose(label_norm.reshape(-1, h // 2, w // 2, 4), (0, 3, 1, 2))).float()  # 变成网络所需要的格式
                        # image :torch.Size([1, 4, 868, 1156]) <class 'torch.Tensor'>

                    x, y = image,label
                    model.to(device2)
                    logits = model(x)  # 数据放入网络中
                    loss = loss_function(logits, y)  # 得到损失值
                    loss_val_all += loss.item() * x.size(0)
            loss_val_av = loss_val_all / 10
            loss_list.append((loss_val_av))
            # if (epoch + 1) == 5:
            #     best_loss = loss_val_av
            print(f'第{epoch + 1}轮，val损失值为：{loss_val_av}')
            if best_loss > loss_val_av:
                    best_loss = loss_val_av
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), 'model_val_100_20.pth')

        e_time = time.time()
        need_one_epoch_time = (e_time - s_time) / 60
        print(f'第{epoch + 1}轮，运行时间为：{need_one_epoch_time} min')
    # if (epoch + 1) == epochs:
    #     torch.save(model.state_dict(), 'model_last_4x100_100.pth')
    print('best_loss:', best_loss, 'best_epoch:', best_epoch)
    return loss_list

def main():
    start_time = time.time()

    lr =5e-4
    epochs = 20

    device1 = torch.device('cuda:0')
    device2 = torch.device('cpu')


    model = Unet()
    model.load_state_dict(torch.load('model.pth'))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.MSELoss()

    loss_list=train(model,optimizer,criteon,epochs,device2,device2)
    end_time = time.time()
    all_time = (end_time - start_time) / 3600
    print(f'所用时间：{all_time} h')
    print('========================================================================')
    print('loss_list:',loss_list)

if __name__ == '__main__':
    main()
