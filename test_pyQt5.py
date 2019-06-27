import sys
from PyQt5.QtWidgets import QApplication, QWidget,QPushButton,QLabel,QFileDialog,QLineEdit
from PyQt5.QtGui import QIcon,QPixmap

from get_result import get_result_dd

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Welcome FER application-VIPLab-Soongsil University'
        self.left = 20
        self.top = 20
        self.width = 640
        self.height = 480
        self.initUI()
        self.value = 'No image choose'

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setFixedSize(640, 480)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label1 = QLabel(self)
        self.label1.setText("Please choose the image want classify. Then click button Classifier to get result\n Facial Expression: Anger,Contempt, Disgust, Fear, Happy, Sadnees, Surprise")
        self.label1.move(50 , 10)

        btn = QPushButton('Choose input image', self)
        btn.resize(btn.sizeHint())
        btn.move(50, 50)
        btn.clicked.connect(self.getfile)

        btn_gen = QPushButton('Classifier', self)
        btn_gen.resize(btn_gen.sizeHint())
        btn_gen.move(200, 50)
        btn_gen.clicked.connect(self.show_content)

        self.textbox = QLineEdit(self)
        self.textbox.move(200, 150)
        self.textbox.resize(150, 30)

        self.label = QLabel(self)
        pixmap = QPixmap('1.png')
        self.label.setPixmap(pixmap)
        self.label.move(50, 150)
        self.resize(pixmap.width(), pixmap.height())
        self.show()

    def getfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            './', "Image files (*.jpg *.gif *.png)")
        print(fname[0])
        self.value = fname[0]
        self.pixmap = QPixmap(fname[0])
        self.label.setPixmap(QPixmap(fname[0]))

    def show_content(self):
        textboxValue = self.textbox.text()
        d = get_result_dd(self.value)
        self.textbox.setText(d)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())