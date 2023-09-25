'''
错误：1.数据和标签反了
     2.训练标准化和最后inference标准化不一致！！！
注意：以一定要一致！！！
速度慢的原因：没有使用batch_size，设备内存不够

数据增强：单独拿出噪声翻倍后变成噪声图，或者两张照片相加
'''
import rawpy
import numpy as np
import torch
import skimage.metrics
import time,utile,random
from torch import nn,optim
#from network import Unet
from network import Unet
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
def get_part_img(data,label,h,w,part):
    data_img, label_img = [], []
    for i in range(part):
        for j in range(part):
            data_img.append(data[i*h:(i+1)*h,j*w:(j+1)*w])
            label_img.append(label[i*h:(i+1)*h,j*w:(j+1)*w])
    return data_img,label_img


def data_aug(input_data, input_label,ex_rand):
    raw_label = input_label
    raw_data = input_data

    black_level, white_level = 1024, 16383

    label = utile.normalization(raw_label, black_level, white_level)
    data = utile.normalization(raw_data, black_level, white_level)

    noise = data - label
    if ex_rand == 0:
        ex_rand = 1.0 + 0.5* random.random()
    if ex_rand != 0 :
        noise *= ex_rand
        data += noise
        data= read_image(data)
        height, width = 3472, 4624
        data = utile.inv_normalization(data, black_level, white_level)
        data = utile.write_image_1(data, height, width)
        #utile.write_back_dng('dataset/ground_truth/0_gt.dng','0_noise_new.dng',data)
    if ex_rand == 0 :
        print('=========================================error=========================================')
    return data, raw_label,ex_rand


#反过来去训练得出噪声
def train(model,optimizer,scheduler,loss_function,epochs,device):
    [w_p, psnr_max, psnr_min, ssim_min] = [0.8, 60, 30, 0.8]
    score_best = 48
    best_psnr, best_ssim,best_epoch = -1, 0.91,0  # 输出验证集中准确率最高的轮次和准确率
    best_t_loss, best_t_epoch = 1, 0
    score_list = []

    for epoch in range(epochs):
        print('=========================================')
        print('==================第{}轮=================='.format(epoch + 1))
        s_time = time.time()
        loss_all = 0
        for i in range(98):  #0-97
            one_s_time =time.time()
            #训练得到噪声
            raw_data = rawpy.imread('dataset/ground_truth/{}_gt.dng'.format(i)).raw_image_visible
            raw_label =rawpy.imread('dataset/noisy/{}_noise.dng'.format(i)).raw_image_visible
            #加入噪声
            if (epoch+1) % 2 == 0:
                if i < 70:
                    raw_label, raw_data, rand_ex = data_aug(raw_label, raw_data, ex_rand=1.2)
                if i >= 70 and i <90:
                    raw_label, raw_data, rand_ex = data_aug(raw_label, raw_data, ex_rand=1.3)
                else:
                    raw_label, raw_data, rand_ex = data_aug(raw_label, raw_data, ex_rand=0)
            else:
                if i >= 80 and i < 90:
                    raw_label,raw_data,rand_ex = data_aug(raw_label,raw_data,ex_rand=1.1) #输入：data，label,输出：data,label
                if i >= 90 and i < 97:
                    raw_label, raw_data, rand_ex = data_aug(raw_label, raw_data, ex_rand=1.2)
                if i == 97 :
                    raw_label, raw_data, rand_ex = data_aug(raw_label, raw_data, ex_rand=1.4)

            h = 434 * 2
            w = 578 * 2

            data_img, label_img = get_part_img(raw_data, raw_label, h, w, 4)

            black_level, white_level = 1024, 16383
            for j in range(16):
                #print(type(raw))#<class 'numpy.ndarray'>
                # h = 434*2
                # w = 578*2
                #print(h,w)
                #1.得到图片和标签数据,使用read_image函数
                #标准化：1024, 16383

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

                x,y = image.to(device),label.to(device)
                logits = model(x)  # 数据放入网络中
                loss = loss_function(logits, y)  # 得到损失值
                loss_all += loss.item() * x.size(0)
                optimizer.zero_grad()  # 优化器先清零，不然会叠加上次的数值
                loss.sum().backward()  # 后向传播
                optimizer.step()
            #测试专用：
            # one_e_time = time.time()
            # print(f'第{epoch+1}轮,完成第{i+1}张图片训练，所用时间：{one_e_time - one_s_time} s')
        loss_av = loss_all / 1568
        print(f'第{epoch + 1}轮，train损失值为：{loss_av}')
        scheduler.step()
        if (epoch+1) % 5 == 0:
            psnr_all,ssim_all =0,0
            model.eval()
            with torch.no_grad():
                for j in range(98,100):  #96-99
                    raw_label = rawpy.imread('dataset/ground_truth/{}_gt.dng'.format(j)).raw_image_visible
                    raw_data = rawpy.imread('dataset/noisy/{}_noise.dng'.format(j)).raw_image_visible

                    if j == 99:
                        raw_data,raw_label,rand = data_aug(raw_data, raw_label,ex_rand=1.20)   #输入：data，label,输出：data,label

                    h = 434 * 4
                    w = 578 * 4
                    data_img_val, label_img_val = get_part_img(raw_data,raw_label,h, w,2)
                    black_level, white_level = 1024, 16383
                    one_val_time = time.time()
                    for i in range(4):
                        img = read_image(data_img_val[i])
                        label = read_image(label_img_val[i])

                        data_nor = normalization(img, black_level, white_level)
                        image = torch.from_numpy(
                                    np.transpose(data_nor.reshape(-1, h // 2, w // 2, 4), (0, 3, 1, 2))).float()

                        label_norm = normalization(label,black_level, white_level)  # 标准化
                        label = torch.from_numpy(
                                        np.transpose(label_norm.reshape(-1, h // 2, w // 2, 4), (0, 3, 1, 2))).float()  # 变成网络所需要的格式
                                    # image :torch.Size([1, 4, 868, 1156]) <class 'torch.Tensor'>

                        x, y = image.to(device),label.to(device)
                        #model.to(device2)
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
                    end_val_time = time.time()
                    time_val = end_val_time - one_val_time
                    print(f'第{epoch+1}轮,完成第{j}张图片检测，所用时间：{time_val} s')
                psnr = psnr_all / 8
                ssim = ssim_all / 8
                print('=========================================')
                print(f'第{epoch + 1}轮，psnr：{psnr},ssim:{ssim}')
                score = (w_p * max(psnr - psnr_min, 0) / (psnr_max - psnr_min) + (1 - w_p) * max(ssim - ssim_min, 0) / (
                            1 - ssim_min)) * 100
                print('打分：',score)
                score_list.append(score)

                if score > score_best:
                        score_best = score
                        best_ssim = ssim
                        best_psnr = psnr
                        best_epoch = epoch + 1
                        torch.save(model.state_dict(), 'model_val_350.pth')
                print('=========================================')
        # if (epoch+1) == 70:
        #     torch.save(model.state_dict(), 'model_val_50.pth')
        if (epoch + 1) == 120:
            torch.save(model.state_dict(), 'model_val_1200.pth')
        if score_best > 48:
            print('当前最高分数：', score_best)
        e_time = time.time()
        need_one_epoch_time = (e_time - s_time) / 60
        print(f'第{epoch + 1}轮，运行时间为：{need_one_epoch_time} min')
        # if (epoch + 1) == 100:
        #     torch.save(model.state_dict(), 'model_val_100.pth')
    print('best_psnr:', best_psnr, 'best_ssim',best_ssim,'best_epoch:', best_epoch,'best_score:',score_best)
    return score_list

def main():
    start_time = time.time()

    lr =1e-4
    epochs = 800

    device = torch.device('cuda:0')
   # device2 = torch.device('cpu')
    model = Unet().to(device)
    model.load_state_dict(torch.load('model_0.pth'))
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,120,180,250,320,550,600,700], 0.7)  #,
    criteon = nn.MSELoss()

    loss_list=train(model,optimizer,scheduler,criteon,epochs,device)
    end_time = time.time()
    all_time = (end_time - start_time) / 3600
    print(f'所用时间：{all_time} h')
    print('========================================================================')
    print('loss_list:',loss_list)

if __name__ == '__main__':
    main()
