"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import AtrousConvolution2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate,merge
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

from ssd_300module2.utils.ssd_layers import Normalize
from ssd_300module2.utils.ssd_layers import PriorBox


def SSD300(input_shape, num_classes=21):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    net = {}
    # Block 1
    # 输入图片的形状tensor
    input_tensor = input_tensor = Input(shape=input_shape)
    # 图片的形状
    img_size = (input_shape[1], input_shape[0])
    # 输入层，输入图片的数据
    net['input'] = input_tensor
    # 卷积层conv1_1(None,300,300,64)
    net['conv1_1'] = Conv2D(64, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv1_1')(net['input'])
    # 卷积层conv1_2(None,300,300,64)
    net['conv1_2'] = Conv2D(64, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv1_2')(net['conv1_1'])
    # 卷积层conv1_2(None,150,150,64)
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool1')(net['conv1_2'])
    # Block 2
    # conv2_1(None,150,150,128)
    net['conv2_1'] = Conv2D(128, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv2_1')(net['pool1'])
    # conv2_2(None,150,150,128)
    net['conv2_2'] = Conv2D(128, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv2_2')(net['conv2_1'])
    # pool2(None,75,75,128)
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool2')(net['conv2_2'])
    # Block 3
    # conv3_1(None,75,75,256)
    net['conv3_1'] = Conv2D(256, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv3_1')(net['pool2'])
    # conv3_2(None,75,75,256)
    net['conv3_2'] = Conv2D(256, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv3_2')(net['conv3_1'])
    # conv3_3(None,75,75,256)
    net['conv3_3'] = Conv2D(256, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv3_3')(net['conv3_2'])
    # pool3(None,38,38,256)
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool3')(net['conv3_3'])
    # Block 4
    # conv4_1(None,38,38,512)
    net['conv4_1'] = Conv2D(512,kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv4_1')(net['pool3'])
    # conv4_2(None,38,38,512)
    net['conv4_2'] = Conv2D(512, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv4_2')(net['conv4_1'])
    # conv4_3(None,38,38,512)
    net['conv4_3'] = Conv2D(512,kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv4_3')(net['conv4_2'])
    # pool4(None,19,19,512)
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool4')(net['conv4_3'])
    # Block 5
    # conv5_1(None,19,19,512)
    net['conv5_1'] = Conv2D(512, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv5_1')(net['pool4'])
    # conv5_2(None,19,19,512)
    # nb_row: Number of rows in the convolution kernel.
    # nb_col: Number of columns in the convolution kerne
    # conv5_2(None,19,19,512)
    net['conv5_2'] = Conv2D(512, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv5_2')(net['conv5_1'])
    # conv5_3(None,19,19,512)
    net['conv5_3'] = Conv2D(512, kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv5_3')(net['conv5_2'])
    # pool5(None,19,19,512)
    net['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                name='pool5')(net['conv5_3'])
    # FC6
    # fc6(None,19,19,1024), 全连接层,膨胀层空洞层Atrous Convolution能够保证这样的池化后的感受野不变，从而可以fine tune，同时也能保证输出的结果更加精细
    net['fc6'] = Conv2D(1024,kernel_size=3, dilation_rate=(6, 6),
                                     activation='relu', padding='same',
                                     name='fc6')(net['pool5'])
    # 增加dropout层
    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    # fc7(None,19,19,1024)
    net['fc7'] = Conv2D(1024, kernel_size=1, activation='relu',
                               padding='same', name='fc7')(net['fc6'])
    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    # conv6_1(None, 19, 19, 256)
    net['conv6_1'] = Conv2D(256,kernel_size=1, activation='relu',
                                   padding='same',
                                   name='conv6_1')(net['fc7'])
    # conv6_2(None, 10, 10, 512),subsample ：strides :指定步长
    net['conv6_2'] = Conv2D(512, kernel_size=3, strides=(2, 2),
                                   activation='relu', padding='same',
                                   name='conv6_2')(net['conv6_1'])
    # Block 7
    # conv7_1(None, 10, 10, 128)
    net['conv7_1'] = Conv2D(128, kernel_size=1, activation='relu',
                                   padding='same',
                                   name='conv7_1')(net['conv6_2'])
    # conv7_2(None, 12, 12, 128) 零填充
    net['conv7_2'] = ZeroPadding2D()(net['conv7_1'])
    # conv7_2(None, 5, 5, 256) 零填充
    net['conv7_2'] = Conv2D(256, kernel_size=3, strides=(2, 2),
                                   activation='relu', padding='valid',
                                   name='conv7_2')(net['conv7_2'])
    # Block 8
    # conv8_1(None, 5, 5, 128)
    net['conv8_1'] = Conv2D(128, kernel_size=1, activation='relu',
                                   padding='same',
                                   name='conv8_1')(net['conv7_2'])
    # conv8_2(None, 3, 3, 256)
    net['conv8_2'] = Conv2D(256, kernel_size=3, strides=(2, 2),
                                   activation='relu', padding='same',
                                   name='conv8_2')(net['conv8_1'])
    # Last Pool
    # pool6(None, 256)
    net['pool6'] = GlobalAveragePooling2D(name='pool6')(net['conv8_2'])


    # Prediction from conv4_3
    # conv4_3_norm(None, 38，38，512)，标准化处理归一化
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    num_priors = 3 # 指定每个像素点提取多少个特征
    # Conv2D(None, 38，38，512)
    x = Conv2D(num_priors * 4, kernel_size=3, padding='same',
                      name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    # 获取的特征feature_map
    net['conv4_3_norm_mbox_loc'] = x
    # 定义一个flatten
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    # 改变形状
    net['conv4_3_norm_mbox_loc_flat'] = flatten(net['conv4_3_norm_mbox_loc'])
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=3, padding='same',
                      name=name)(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf'] = x
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    net['conv4_3_norm_mbox_conf_flat'] = flatten(net['conv4_3_norm_mbox_conf'])
    # 生成预设的anchors
    #以feature map上每个点的中点为中心（offset=0.5），生成一些列同心的prior box（然后中心点的坐标会乘以step，相当于从feature map位置映射回原图位置）
    # img_size:图片的尺寸
    # min_size:正方形prior box最小边长为min_size，最大边长为：squres(min_size*max_size)
    # aspect_rati0s:每在prototxt设置一个aspect ratio，会生成2个长方形
    # variances:4个variance实际上是一种bounding regression中的权重
    # 而每个feature map对应prior box的min_size和max_size由以下公式决定，公式中m是使用feature map的数量（SSD 300中m=6）
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])


    # Prediction from fc7
    num_priors = 6
    net['fc7_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=3,
                                        padding='same',
                                        name='fc7_mbox_loc')(net['fc7'])
    flatten = Flatten(name='fc7_mbox_loc_flat')
    net['fc7_mbox_loc_flat'] = flatten(net['fc7_mbox_loc'])
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes,kernel_size=3,
                                         padding='same',
                                         name=name)(net['fc7'])
    flatten = Flatten(name='fc7_mbox_conf_flat')
    net['fc7_mbox_conf_flat'] = flatten(net['fc7_mbox_conf'])
    priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])


    # Prediction from conv6_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=3, padding='same',
                      name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc'] = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    net['conv6_2_mbox_loc_flat'] = flatten(net['conv6_2_mbox_loc'])
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=3, padding='same',
                      name=name)(net['conv6_2'])
    net['conv6_2_mbox_conf'] = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    net['conv6_2_mbox_conf_flat'] = flatten(net['conv6_2_mbox_conf'])
    priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])


    # Prediction from conv7_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=3, padding='same',
                      name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc'] = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    net['conv7_2_mbox_loc_flat'] = flatten(net['conv7_2_mbox_loc'])
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=3, padding='same',
                      name=name)(net['conv7_2'])
    net['conv7_2_mbox_conf'] = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    net['conv7_2_mbox_conf_flat'] = flatten(net['conv7_2_mbox_conf'])
    priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])

    # Prediction from conv8_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=3, padding='same',
                      name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc'] = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    net['conv8_2_mbox_loc_flat'] = flatten(net['conv8_2_mbox_loc'])
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=3, padding='same',
                      name=name)(net['conv8_2'])
    net['conv8_2_mbox_conf'] = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    net['conv8_2_mbox_conf_flat'] = flatten(net['conv8_2_mbox_conf'])
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')

    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])

    # Prediction from pool6
    num_priors = 6
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    net['pool6_mbox_loc_flat'] = x
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Dense(num_priors * num_classes, name=name)(net['pool6'])
    net['pool6_mbox_conf_flat'] = x
    priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')
    if K.image_data_format() == 'channels_last':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    net['pool6_reshaped'] = Reshape(target_shape,
                                    name='pool6_reshaped')(net['pool6'])
    net['pool6_mbox_priorbox'] = priorbox(net['pool6_reshaped'])
    # Gather all predictions
    # 位置合并
    net['mbox_loc'] = concatenate([net['conv4_3_norm_mbox_loc_flat'],
                             net['fc7_mbox_loc_flat'],
                             net['conv6_2_mbox_loc_flat'],
                             net['conv7_2_mbox_loc_flat'],
                             net['conv8_2_mbox_loc_flat'],
                             net['pool6_mbox_loc_flat']],
                             axis=1, name='mbox_loc')
    # 置信度合并
    net['mbox_conf'] = concatenate([net['conv4_3_norm_mbox_conf_flat'],
                              net['fc7_mbox_conf_flat'],
                              net['conv6_2_mbox_conf_flat'],
                              net['conv7_2_mbox_conf_flat'],
                              net['conv8_2_mbox_conf_flat'],
                              net['pool6_mbox_conf_flat']],
                              axis=1, name='mbox_conf')
    # 网络层合并选出的盒子层
    net['mbox_priorbox'] = concatenate([net['conv4_3_norm_mbox_priorbox'],
                                  net['fc7_mbox_priorbox'],
                                  net['conv6_2_mbox_priorbox'],
                                  net['conv7_2_mbox_priorbox'],
                                  net['conv8_2_mbox_priorbox'],
                                  net['pool6_mbox_priorbox']],
                                  axis=1,name='mbox_priorbox')

    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4
    # 重设形状
    net['mbox_loc'] = Reshape((num_boxes, 4),
                              name='mbox_loc_final')(net['mbox_loc'])
    # 重设形状
    net['mbox_conf'] = Reshape((num_boxes, num_classes),
                               name='mbox_conf_logits')(net['mbox_conf'])
    # 盒子置信度使用softmax激活函数
    net['mbox_conf'] = Activation('softmax',
                                  name='mbox_conf_final')(net['mbox_conf'])
    # 将盒子的位置，盒子的置信度，盒子的特征合并
    net['predictions'] = concatenate([net['mbox_loc'],
                               net['mbox_conf'],
                               net['mbox_priorbox']],
                               axis=2,name='predictions')
    # 打印盒子的位置，盒子的置信度，priorbox获取的盒子
    print(net['mbox_loc'], net['mbox_conf'], net['mbox_priorbox'])
    # 获取最终定义的模型
    model = Model(net['input'], net['predictions'])
    return model
