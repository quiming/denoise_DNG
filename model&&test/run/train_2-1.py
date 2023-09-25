'''
错误：1.数据和标签反了
     2.训练标准化和最后inference标准化不一致！！！
注意：以一定要一致！！！
'''
import rawpy
import numpy as np
import torch
import skimage.metrics
import time
from torch import nn,optim
#from network import Unet
from network1 import Unet
from utile import inv_normalization,write_image

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
def train(model,optimizer,scheduler,loss_function,epochs,device1,device2):
    [w_p, psnr_max, psnr_min, ssim_min] = [0.8, 60, 30, 0.8]
    score_best = 50
    best_psnr, best_ssim,best_epoch = -1, 0.91,0  # 输出验证集中准确率最高的轮次和准确率
    best_t_loss, best_t_epoch = 1, 0
    loss_list = []

    for epoch in range(epochs):
        print('=========================================')
        print('==================第{}轮=================='.format(epoch + 1))
        s_time = time.time()
        loss_all = 0
        for i in range(96):  #0-95
            #训练得到噪声
            raw_data = rawpy.imread('dataset/ground_truth/{}_gt.dng'.format(i)).raw_image_visible
            raw_label =rawpy.imread('dataset/noisy/{}_noise.dng'.format(i)).raw_image_visible
            #one_s_time = time.time()
            for j in range(16):
                data_img,label_img = [],[]
                #print(type(raw))#<class 'numpy.ndarray'>
                h = 868
                w = 1156
                #print(h,w)
                #1.得到图片和标签数据,使用read_image函数
                data_img.append(raw_data[:h, :w]), data_img.append(raw_data[h:2 * h, :w]), data_img.append(
                    raw_data[2 * h:3 * h, :w]), data_img.append(raw_data[3 * h:, :w]),
                data_img.append(raw_data[:h, w:2 * w]), data_img.append(raw_data[h:2 * h, w:2 * w]), data_img.append(
                    raw_data[2 * h:3 * h, w:2 * w]), data_img.append(raw_data[3 * h:, w:2 * w]),
                data_img.append(raw_data[:h, 2 * w:3 * w]), data_img.append(
                    raw_data[h:2 * h, 2 * w:3 * w]), data_img.append(
                    raw_data[2 * h:3 * h, 2 * w:3 * w]), data_img.append(raw_data[3 * h:, 2 * w:3 * w]),
                data_img.append(raw_data[:h, 3 * w:]), data_img.append(raw_data[h:2 * h, 3 * w:]), data_img.append(
                    raw_data[2 * h:3 * h, 3 * w:]), data_img.append(raw_data[3 * h:, 3 * w:]),

                label_img.append(raw_label[:h, :w]), label_img.append(raw_label[h:2 * h, :w]), label_img.append(
                    raw_label[2 * h:3 * h, :w]), label_img.append(raw_label[3 * h:, :w])
                label_img.append(raw_label[:h, w:2 * w]), label_img.append(
                    raw_label[h:2 * h, w:2 * w]), label_img.append(raw_label[2 * h:3 * h, w:2 * w]), label_img.append(
                    raw_label[3 * h:, w:2 * w]),
                label_img.append(raw_label[:h, 2 * w:3 * w]), label_img.append(
                    raw_label[h:2 * h, 2 * w:3 * w]), label_img.append(
                    raw_label[2 * h:3 * h, 2 * w:3 * w]), label_img.append(raw_label[3 * h:, 2 * w:3 * w]),
                label_img.append(raw_label[:h, 3 * w:]), label_img.append(raw_label[h:2 * h, 3 * w:]), label_img.append(
                    raw_label[2 * h:3 * h, 3 * w:]), label_img.append(raw_label[3 * h:, 3 * w:]),

                #标准化：1024, 16383
                black_level, white_level = 1024, 16383
                #训练得到噪声
                #noise - gt     noise - noise +gt =gt
                label_img[j] = read_image(label_img[j])
                data_nor = normalization(label_img[j] ,black_level, white_level)
                image = torch.from_numpy(np.transpose(data_nor.reshape(-1, h//2, w//2, 4), (0, 3, 1, 2))).float()   #image: torch.Size([1, 4, 868, 1156])

                data_img[j] = read_image(data_img[j])
                label_norm = normalization(data_img[j],black_level, white_level )  #标准化
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
            #one_e_time = time.time()
            #print(f'第{epoch+1}轮,完成第{i+1}张图片训练，所用时间：{one_e_time - one_s_time} s')
        loss_av = loss_all / 1536
        print(f'第{epoch + 1}轮，train损失值为：{loss_av}')
        # # if epoch == 1:
        # #     best_t_loss = loss_av
        if (epoch + 1) % 2 == 0:
            if best_t_loss > loss_av:
                best_t_loss = loss_av
                best_t_epoch = epoch + 1
                torch.save(model.state_dict(), 'model_train_4x100_500.pth')
        scheduler.step()
        if (epoch+1) % 4 == 0:
            #psnr_ls,ssim_ls = [],[]
            psnr_all,ssim_all =0,0
            model.eval()
            with torch.no_grad():
                for i in range(96,100):  #96-99
                    raw_label = rawpy.imread('dataset/ground_truth/{}_gt.dng'.format(i)).raw_image_visible
                    raw_data = rawpy.imread('dataset/noisy/{}_noise.dng'.format(i)).raw_image_visible

                    img = read_image(raw_data)
                    label = read_image(raw_label)
                    black_level, white_level = 1024, 16383
                    h = 3472
                    w = 4624
                            # noise - gt     noise - noise +gt =gt
                    data_nor = normalization(img, black_level, white_level)
                    image = torch.from_numpy(
                                np.transpose(data_nor.reshape(-1, h // 2, w // 2, 4), (0, 3, 1, 2))).float()

                    label_norm = normalization(label,black_level, white_level)  # 标准化
                                #label_norm = data_nor - label_norm
                    label = torch.from_numpy(
                                    np.transpose(label_norm.reshape(-1, h // 2, w // 2, 4), (0, 3, 1, 2))).float()  # 变成网络所需要的格式
                                # image :torch.Size([1, 4, 868, 1156]) <class 'torch.Tensor'>

                    x, y = image,label
                    model.to(device2)
                    logits = model(x)  # 数据放入网络中
                    result_data = logits.cpu().detach().numpy().transpose(0, 2, 3, 1)
                    result_data = result_data.reshape(-1, h // 2, w // 2, 4)
                    result_data = inv_normalization(result_data, black_level, white_level)
                    result_write_data = write_image(result_data, h, w)

                    gt = label.cpu().detach().numpy().transpose(0, 2, 3, 1)
                    gt = gt.reshape(-1, h // 2, w // 2, 4)
                    gt = inv_normalization(gt, black_level, white_level)
                    gt = write_image(gt, h, w)

                    psnr = skimage.metrics.peak_signal_noise_ratio(
                                gt.astype(np.float), result_write_data.astype(np.float), data_range=white_level)
                    psnr_all += psnr
                    ssim = skimage.metrics.structural_similarity(
                                gt.astype(np.float), result_write_data.astype(np.float), multichannel=True,
                                data_range=white_level)
                    ssim_all += ssim
                psnr = psnr_all / 4
                ssim = ssim_all / 4
                print('=========================================')
                print(f'第{epoch + 1}轮，psnr：{psnr},ssim:{ssim}')
                score = (w_p * max(psnr - psnr_min, 0) / (psnr_max - psnr_min) + (1 - w_p) * max(ssim - ssim_min, 0) / (
                            1 - ssim_min)) * 100
                print('打分：',score)
                if score > score_best:
                        score_best = score
                        best_ssim = ssim
                        best_psnr = psnr
                        best_epoch = epoch + 1
                        torch.save(model.state_dict(), 'model_val_500.pth')
                print('=========================================')
        e_time = time.time()
        need_one_epoch_time = (e_time - s_time) / 60
        print(f'第{epoch + 1}轮，运行时间为：{need_one_epoch_time} min')
    # if (epoch + 1) == epochs:
    #     torch.save(model.state_dict(), 'model_last_4x100_100.pth')
    print('best_psnr:', best_psnr, 'best_ssim',best_ssim,'best_epoch:', best_epoch,'best_score:',score_best)
    return loss_list

def main():
    start_time = time.time()

    lr =1e-4
    epochs = 320

    device1 = torch.device('cuda:0')
    device2 = torch.device('cpu')


    model = Unet()
    model.load_state_dict(torch.load('model_val.pth'))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [32,82,150,220,300], 0.6)   #32,82,
    criteon = nn.MSELoss()

    loss_list=train(model,optimizer,scheduler,criteon,epochs,device1,device2)
    end_time = time.time()
    all_time = (end_time - start_time) / 3600
    print(f'所用时间：{all_time} h')
    print('========================================================================')
    print('loss_list:',loss_list)

if __name__ == '__main__':
    main()
