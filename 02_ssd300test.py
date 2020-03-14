"""
定义好类别数量以及输出
模型预测流程
SSD300模型输入以及加载参数
读取多个本地路径测试图片，preprocess_input以及保存图像像素值（显示需要）
模型预测结果，得到7308个priorbox
进行非最大抑制算法处理
SSD300模型输入以及加载参数
"""

from tensorflow.python.keras.preprocessing.image import img_to_array,load_img
# imread将ndarray数据转为二进制数据可以进行绘图装作
from imageio import imread # 读取变为array
from keras.applications.imagenet_utils import preprocess_input # 图片进行预处理
from ssd_300module2.nets.ssd_net import SSD300
import numpy as np
import os
from ssd_300module2.utils.ssd_utils import BBoxUtility
import matplotlib.pyplot as plt



class SSDTest(object):
    def __init__(self):
        self.classes_name = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                             'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                             'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                             'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        self.classes_nums = len(self.classes_name) + 1
        self.input_shape = (300, 300, 3)

    def test(self):
        # SSD300模型输入以及加载参数
        model = SSD300(self.input_shape, num_classes=self.classes_nums)
        model.load_weights(r"./ckpt/pre_trained/weights_SSD300.hdf5",by_name=True)
        # 读取多个本地路径测试图片，preprocess_input以及保存图像像素值（显示需要）
        # 循环读取图片进行多个图片输出检测
        feature = []
        images = []
        for pic_name in os.listdir("./image/"):
            img_path = os.path.join("./image/", pic_name)
            image = load_img(img_path,target_size=(self.input_shape[0],self.input_shape[1]))
            image = img_to_array(image)
            feature.append(image)

            # 保存原始图片的数组,二进制数据
            images.append(imread(img_path))

        # 处理图片数据,ndarray数组输入
        # 将列表转为相应的数组,归一化处理，标准化处理
        inputs = preprocess_input(np.array(feature))

        # 预测,批量预测(2, 7308, 33)
        y_predict = model.predict(inputs)

        # 模型预测结果，得到7308个priorbox
        # 进行非最大抑制算法处理NMS 21
        bb = BBoxUtility(self.classes_nums)
        # 进行解析(200, 6)
        # 身下200个候选框，类别保留6个
        prior_box = bb.detection_out(y_predict)
        print(prior_box[0].shape,prior_box[1].shape)
        return prior_box,images

    def tag_picture(self, images, outputs):
        """
        对图片预测结果画图显示
        :param images:
        :param outputs:
        :return:
        """

        # 解析输出结果,每张图片的标签，置信度和位置
        for i,img in enumerate(images):
            # 通过i获取图片label, location, xmin, ymin, xmax, ymax
            pre_label = outputs[i][:, 0]
            pre_conf = outputs[i][:, 1]
            pre_xmin = outputs[i][:, 2]
            pre_ymin = outputs[i][:, 3]
            pre_xmax = outputs[i][:, 4]
            pre_ymax = outputs[i][:, 5]
            # print("label:{}, probability:{}, xmin:{}, ymin:{}, xmax:{}, ymax:{}".
            #       format(pre_label, pre_conf, pre_xmin, pre_ymin, pre_xmax, pre_ymax))

            # 过滤预测框到指定类别的概率小的 prior box
            top_indices = [i for i,conf in enumerate(pre_conf) if conf > 0.6]
            print(top_indices)
            top_conf = pre_conf[top_indices]
            # tolist 将ndarray 转为list
            top_label_indices = pre_label[top_indices].tolist()
            top_xmin = pre_xmin[top_indices]
            top_ymin = pre_ymin[top_indices]
            top_xmax = pre_xmax[top_indices]
            top_ymax = pre_ymax[top_indices]

            print("label:{}, probability:{}, xmin:{}, ymin:{}, xmax:{}, ymax:{}".
                  format(top_label_indices, top_conf, top_xmin, top_ymin, top_xmax, top_ymax))

            # matplotlib图片显示
            # 定义21中颜色，显示图片
            # currentAxis增加图中文本显示和标记显示 准备0，1之间的数字，通过hsv转为颜色
            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            plt.imshow(img / 255.)  # plt.imshow(img/255.0) #取值范围为0.0到1.0的浮点数组，能正确显示颜色
            currentAxis = plt.gca() # Get Current Axes获取当前图形的坐标系

            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * img.shape[1])) # 实际图片的宽，保留整数
                ymin = int(round(top_ymin[i] * img.shape[0])) # 实际图片的长
                xmax = int(round(top_xmax[i] * img.shape[1])) # 实际图片的宽
                ymax = int(round(top_ymax[i] * img.shape[0])) # 实际图片的长

                # 获取该图片预测概率，名称，定义显示颜色
                score = top_conf[i]
                # 获取标签的，颜色，
                label = int(top_label_indices[i]) # 标签的需要从1开始
                label_name = self.classes_name[label - 1] # 标签的名字

                # 显示文本的格式
                display_txt = '{:0.2f}, {}'.format(score, label_name)
                # 坐标
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                # 指定类型的颜色
                color = colors[label]
                # 显示方框，通过获取的坐标系绘制图形
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                # 左上角显示概率以及名称
                currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

            plt.show()

if __name__ == '__main__':
    ssd = SSDTest()
    prior_bbox,image_data = ssd.test()
    ssd.tag_picture(image_data,prior_bbox)