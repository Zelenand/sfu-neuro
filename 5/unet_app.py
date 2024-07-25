from gui import GUI
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
import torch
import torchvision

from unet_model import UNet, UNet2, UNet3


class UNetApplication(QDialog, GUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.open_image)
        self.pushButton_2.clicked.connect(self.process_image)

        self.model = UNet()
        self.model.load_state_dict(torch.load('model_weights.pth'))
        self.model.eval()
        self.model_2 = UNet2()
        self.model_2.load_state_dict(torch.load('model_weights_2.pth'))
        self.model_2.eval()
        self.model_3 = UNet3()
        self.model_3.load_state_dict(torch.load('model_weights_3.pth'))
        self.model_3.eval()
        self.image_path = None

    def open_image(self):
        try:
            self.image_path = QFileDialog.getOpenFileName(self, 'Open File', os.getcwd().replace('\\', '/'), 'Image (*.png *.jpg *.bmp)')[0]
            self.label.setText(str(self.image_path))
            scene = QtWidgets.QGraphicsScene(self)
            print(self.image_path)
            pixmap = QPixmap(self.image_path).scaled(256, 256)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.graphicsView.setScene(scene)
        except:
            self.label.setText("Open image error")


    def process_image(self):
        if self.image_path:
            img = imread(self.image_path)
            size = (256, 256)
            img = [resize(x, size, mode='constant', anti_aliasing=True, ) for x in [img]]
            img = np.array(img, np.float32)
            img = np.rollaxis(img, 3, 1)
            img_tensor = torch.from_numpy(img)
            processed_img = img
            if self.comboBox.currentText() == "UNet1":
                processed_img = self.model(img_tensor)
            elif self.comboBox.currentText() == "UNet2":
                processed_img = self.model_2(img_tensor)
            elif self.comboBox.currentText() == "UNet3":
                processed_img = self.model_3(img_tensor)
            torchvision.utils.save_image(processed_img, 'result_image.png')

            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QPixmap('result_image.png').scaled(256, 256)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.graphicsView_2.setScene(scene)