import utile
import rawpy,random
import numpy as np
def read_image(raw_data):
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2) #(3472, 4624,1)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)   #（1736，2312，4）
    #print('raw_data_expand_c ',raw_data_expand_c.shape)
    return raw_data_expand_c, height, width
def write_image(input_data, height, width):  #3472，4624
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[ :, :, 2 * channel_y + channel_x]
    return output_data
def data_aug(input_data,input_label):
    raw_label = rawpy.imread(input_label).raw_image_visible
    raw_data = rawpy.imread(input_data).raw_image_visible

    black_level, white_level = 1024, 16383

    label = utile.normalization(raw_label,black_level, white_level)
    data = utile.normalization(raw_data,black_level, white_level)

    noise = data - label
    rand = 1.4

    noise *= rand
    data += noise
    data,height, width = read_image(data)
    data = utile.inv_normalization(data,black_level, white_level)
    data = write_image(data,height, width)
    utile.write_back_dng('16_noise.dng','16_noise_new.dng',data)
    return data,raw_label

input_label = '16_gt.dng'
input_data = '16_noise.dng'


data, raw_label = data_aug(input_label,input_data)


def data_aug_2(input_data,input_label,transfor_data):
    raw_label = rawpy.imread(input_label).raw_image_visible
    raw_data = rawpy.imread(input_data).raw_image_visible
    t_data = rawpy.imread(transfor_data).raw_image_visible

    black_level, white_level = 1024, 16383

    label = utile.normalization(raw_label,black_level, white_level)
    data = utile.normalization(raw_data,black_level, white_level)
    t_data = utile.normalization(t_data,black_level, white_level)

    noise = data - label
    rand = 1.4

    noise *= rand
    t_data += noise
    data,height, width = read_image(t_data)
    data = utile.inv_normalization(data,black_level, white_level)
    data = write_image(data,height, width)
    utile.write_back_dng('16_noise.dng','16_noise_new.dng',data)
    #return data,raw_label