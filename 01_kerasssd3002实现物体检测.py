# keras = 1.2.2
from ssd_300module2.nets import ssd_net


if __name__ == '__main__':
    model = ssd_net.SSD300((300, 300, 3), num_classes=21)
    # print(model.summary())