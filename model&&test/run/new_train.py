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
def read_image(raw_data):
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    #print(raw_data_expand_c.shape)
    return raw_data_expand_c
#反过来去训练得出噪声
def train(model,optimizer,loss_function,epochs,device1,device2):
    best_loss, best_epoch = 1, 0  # 输出验证集中准确率最高的轮次和准确率
    best_t_loss, best_t_epoch = 1, 0
    loss_list = []
    for epoch in range(epochs):
        print('==================第{}轮=================='.format(epoch + 1))
        s_time = time.time()
        loss_all = 0
        for i in range(99):
            raw_label= rawpy.imread('dataset/ground_truth/{}_gt.dng'.format(i)).raw_image_visible
            raw_data =rawpy.imread('dataset/noisy/{}_noise.dng'.format(i)).raw_image_visible
            for j in range(16):
                data_img,label_img = [],[]
                #print(type(raw))#<class 'numpy.ndarray'>
                h = 868 #1736
                w = 1156 #2312
                #print(h,w)
                data_img.append(raw_data[:h,:w]),data_img.append(raw_data[h:2*h,:w]),data_img.append(raw_data[2*h:3*h,:w]),data_img.append(raw_data[3*h:,:w]),
                data_img.append(raw_data[:h, w:2*w]), data_img.append(raw_data[h:2 * h,w:2*w]), data_img.append(raw_data[2 * h:3 * h,w:2*w]), data_img.append(raw_data[3 * h:,w:2*w]),
                data_img.append(raw_data[:h, 2*w:3*w]), data_img.append(raw_data[h:2 * h, 2*w:3*w]), data_img.append(raw_data[2 * h:3 * h, 2*w:3*w]), data_img.append(raw_data[3 * h:, 2*w:3*w]),
                data_img.append(raw_data[:h, 3*w:]), data_img.append(raw_data[h:2 * h, 3*w:]), data_img.append( raw_data[2 * h:3 * h, 3*w:]), data_img.append(raw_data[3 * h:, 3*w:]),

                label_img.append(raw_label[:h, :w]),label_img.append(raw_label[h:2*h,:w]),label_img.append(raw_label[2*h:3*h,:w]),label_img.append(raw_label[3*h:,:w])
                label_img.append(raw_label[:h, w:2*w]),label_img.append(raw_label[h:2 * h,w:2*w]),label_img.append(raw_label[2 * h:3 * h,w:2*w]),label_img.append(raw_label[3 * h:,w:2*w]),
                label_img.append(raw_label[:h, 2*w:3*w]),label_img.append(raw_label[h:2 * h, 2*w:3*w]),label_img.append(raw_label[2 * h:3 * h, 2*w:3*w]),label_img.append(raw_label[3 * h:, 2*w:3*w]),
                label_img.append(raw_label[:h, 3*w:]),label_img.append(raw_label[h:2 * h, 3*w:]),label_img.append(raw_label[2 * h:3 * h, 3*w:]),label_img.append(raw_label[3 * h:, 3*w:]),


                black_level, white_level = 1024, 16383
                label_img[j] = read_image(label_img[j])
                label = normalization(label_img[j],black_level, white_level)  #标准化
                label = torch.from_numpy(np.transpose(label.reshape(-1, h//2, w//2, 4), (0, 3, 1, 2))).float()   #变成网络所需要的格式
                # image :torch.Size([1, 4, 868, 1156]) <class 'torch.Tensor'>

                data_img[j] = read_image(data_img[j])
                data = normalization(data_img[j] ,black_level, white_level)
                image = torch.from_numpy(np.transpose(data.reshape(-1, h//2, w//2, 4), (0, 3, 1, 2))).float()

                x,y = image.to(device1),label.to(device1)
                model.to(device1)
                logits = model(x)  # 数据放入网络中
                loss = loss_function(logits, y)  # 得到损失值
                loss_all += loss.item() * x.size(0)
                optimizer.zero_grad()  # 优化器先清零，不然会叠加上次的数值
                loss.backward()  # 后向传播
                optimizer.step()

        loss_av = loss_all / 1584
        print(f'第{epoch + 1}轮，train损失值为：{loss_av}')
        # if epoch == 1:
        #     best_t_loss = loss_av
        if (epoch + 1) % 1 == 0:
            if best_t_loss > loss_av:
                best_t_loss = loss_av
                best_t_epoch = epoch + 1
                torch.save(model.state_dict(), 'model_train.pth')

        if (epoch+1) % 2 ==0:
            loss_val_all = 0
            model.eval()
            for i in range(99,100):
                    raw_label = rawpy.imread('dataset/ground_truth/{}_gt.dng'.format(i)).raw_image_visible
                    raw_data = rawpy.imread('dataset/noisy/{}_noise.dng'.format(i)).raw_image_visible
                    for j in range(16):
                        data_img, label_img = [], []
                        # print(type(raw))#<class 'numpy.ndarray'>
                        h = 868
                        w = 1156
                        # print(h,w)
                        data_img.append(raw_data[:h, :w]), data_img.append(raw_data[h:2 * h, :w]), data_img.append(
                            raw_data[2 * h:3 * h, :w]), data_img.append(raw_data[3 * h:, :w]),
                        data_img.append(raw_data[:h, w:2 * w]), data_img.append(
                            raw_data[h:2 * h, w:2 * w]), data_img.append(
                            raw_data[2 * h:3 * h, w:2 * w]), data_img.append(raw_data[3 * h:, w:2 * w]),
                        data_img.append(raw_data[:h, 2 * w:3 * w]), data_img.append(
                            raw_data[h:2 * h, 2 * w:3 * w]), data_img.append(
                            raw_data[2 * h:3 * h, 2 * w:3 * w]), data_img.append(raw_data[3 * h:, 2 * w:3 * w]),
                        data_img.append(raw_data[:h, 3 * w:]), data_img.append(
                            raw_data[h:2 * h, 3 * w:]), data_img.append(raw_data[2 * h:3 * h, 3 * w:]), data_img.append(
                            raw_data[3 * h:, 3 * w:]),

                        label_img.append(raw_label[:h, :w]), label_img.append(raw_label[h:2 * h, :w]), label_img.append(
                            raw_label[2 * h:3 * h, :w]), label_img.append(raw_label[3 * h:, :w])
                        label_img.append(raw_label[:h, w:2 * w]), label_img.append(
                            raw_label[h:2 * h, w:2 * w]), label_img.append(
                            raw_label[2 * h:3 * h, w:2 * w]), label_img.append(raw_label[3 * h:, w:2 * w]),
                        label_img.append(raw_label[:h, 2 * w:3 * w]), label_img.append(
                            raw_label[h:2 * h, 2 * w:3 * w]), label_img.append(
                            raw_label[2 * h:3 * h, 2 * w:3 * w]), label_img.append(raw_label[3 * h:, 2 * w:3 * w]),
                        label_img.append(raw_label[:h, 3 * w:]), label_img.append(
                            raw_label[h:2 * h, 3 * w:]), label_img.append(
                            raw_label[2 * h:3 * h, 3 * w:]), label_img.append(raw_label[3 * h:, 3 * w:]),

                        black_level, white_level = 1024, 16383
                        data_img[j] = read_image(data_img[j])
                        data = normalization(data_img[j], black_level, white_level)
                        image = torch.from_numpy(
                            np.transpose(data.reshape(-1, h // 2, w // 2, 4), (0, 3, 1, 2))).float()
                        # image :torch.Size([1, 4, 868, 1156]) <class 'torch.Tensor'>
                        label_img[j] = read_image(label_img[j])
                        data_label = normalization(label_img[j], black_level, white_level)
                        label = torch.from_numpy(
                            np.transpose(data_label.reshape(-1, h // 2, w // 2, 4), (0, 3, 1, 2))).float()

                        x, y = image,label
                        model.to(device2)
                        logits = model(x)  # 数据放入网络中
                        loss = loss_function(logits, y)  # 得到损失值
                        loss_val_all += loss.item() * x.size(0)
            loss_val_av = loss_val_all / 16
            loss_list.append((loss_val_av))
            # if (epoch + 1) == 5:
            #     best_loss = loss_val_av
            print(f'第{epoch + 1}轮，val损失值为：{loss_val_av}')
            if best_loss > loss_val_av:
                    best_loss = loss_val_av
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), 'model_val.pth')

        e_time = time.time()
        need_one_epoch_time = (e_time - s_time) / 60
        print(f'第{epoch + 1}轮，运行时间为：{need_one_epoch_time} min')
    # if (epoch + 1) == epochs:
    #     torch.save(model.state_dict(), 'model_last_4x100_100.pth')
    print('best_loss:', best_loss, 'best_epoch:', best_epoch)
    return loss_list

def main():
    start_time = time.time()

    lr =1e-4
    epochs = 30

    device1 = torch.device('cuda:0')
    device2 = torch.device('cpu')

    model = Unet()
    model.load_state_dict(torch.load('model.pth'))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.MSELoss()

    loss_list=train(model,optimizer,criteon,epochs,device1,device2)
    end_time = time.time()
    all_time = (end_time - start_time) / 3600
    print(f'所用时间：{all_time} h')
    print('========================================================================')
    print('loss_list:',loss_list)

if __name__ == '__main__':
    main()
