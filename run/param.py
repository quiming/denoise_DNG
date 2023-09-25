import torch
from network import Unet
net1 = Unet()
net2 = Unet()
out_net= Unet()


net1.load_state_dict(torch.load('model_1.pth'))   #对9好
net2.load_state_dict(torch.load('model_2.pth'))
out_net.load_state_dict(torch.load('model_a.pth'))

param_name = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
              'conv2_1.weight', 'conv2_1.bias', 'conv2_t.weight', 'conv2_t.bias', 'conv2_2.weight', 'conv2_2.bias',
              'conv3.p1_1.weight', 'conv3.p1_1.bias', 'conv3.p2_1.weight', 'conv3.p2_1.bias', 'conv3.p2_2.weight', 'conv3.p2_2.bias', 'conv3.p3_1.weight', 'conv3.p3_1.bias', 'conv3.p3_2.weight', 'conv3.p3_2.bias', 'conv3.p3_3.weight', 'conv3.p3_3.bias', 'conv3.p4_2.weight', 'conv3.p4_2.bias', 'conv4.p1_1.weight', 'conv4.p1_1.bias', 'conv4.p2_1.weight', 'conv4.p2_1.bias', 'conv4.p2_2.weight', 'conv4.p2_2.bias', 'conv4.p3_1.weight', 'conv4.p3_1.bias', 'conv4.p3_2.weight', 'conv4.p3_2.bias', 'conv4.p3_3.weight',
              'conv4.p3_3.bias', 'conv4.p4_2.weight', 'conv4.p4_2.bias',
              'conv5.p1_1.weight', 'conv5.p1_1.bias', 'conv5.p2_1.weight', 'conv5.p2_1.bias', 'conv5.p2_2.weight', 'conv5.p2_2.bias', 'conv5.p3_1.weight', 'conv5.p3_1.bias', 'conv5.p3_2.weight', 'conv5.p3_2.bias', 'conv5.p3_3.weight', 'conv5.p3_3.bias', 'conv5.p4_2.weight', 'conv5.p4_2.bias',
              'upv6.weight', 'upv6.bias', 'conv6_1.weight', 'conv6_1.bias', 'conv6_t.weight', 'conv6_t.bias', 'conv6_2.weight', 'conv6_2.bias', 'upv7.weight', 'upv7.bias', 'conv7_1.weight', 'conv7_1.bias', 'conv7_t.weight', 'conv7_t.bias', 'conv7_2.weight', 'conv7_2.bias', 'upv8.weight', 'upv8.bias', 'conv8_1.weight', 'conv8_1.bias', 'conv8_t.weight', 'conv8_t.bias', 'conv8_2.weight', 'conv8_2.bias', 'upv9.weight', 'upv9.bias', 'conv9_1.weight', 'conv9_1.bias', 'conv9_2.weight', 'conv9_2.bias', 'conv10_1.weight', 'conv10_1.bias']
# # #84
# print(len(param_name))
# for i in range(int(0.5*len(param_name))):
#     net1.state_dict()[param_name[i]] = net2.state_dict()[param_name[i]]

#torch.save(net1.state_dict(),'model_a.pth')
#out_net.state_dict()[param_name[0]] = net1.state_dict()[param_name[0]]+net2.state_dict()[param_name[0]]

print(net1.state_dict()[param_name[0]])

#print(0.5*net1.state_dict()[param_name[0]]+0.5*net2.state_dict()[param_name[0]])




