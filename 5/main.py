from PyQt5.QtWidgets import QApplication
import sys
from unet_app import UNetApplication


if __name__ == '__main__':
    app = QApplication(sys.argv)
    UNet_app = UNetApplication()
    UNet_app.show()
    sys.exit(app.exec_())