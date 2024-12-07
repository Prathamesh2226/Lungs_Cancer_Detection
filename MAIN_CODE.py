
from os import listdir
from os.path import isfile, join

import tkinter
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import PyQt5.QtWidgets as QtWidgets
from tkinter import filedialog
from tkinter import messagebox
from PyQt5 import QtCore, QtGui , QtWidgets
import cv2
from PyQt5.QtGui import QPixmap

import tkinter
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *


import numpy
import numpy as np
import cv2
import pickle
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import itertools

import matplotlib.pyplot as plt

root=tkinter.Tk()

import tkinter
from tkinter import filedialog
from tkinter import messagebox
import cv2
import tensorflow as tf, sys
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
root=tkinter.Tk()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('TRUE CLASS')
    plt.xlabel('PREDICTED CLASS')
    plt.tight_layout()
    
def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
     accum = np.zeros_like(img)
     for kern in filters:
         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
         np.maximum(accum, fimg, accum)
     return accum
    
      
print('******Start*****')
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

class Ui_MainWindow1(object):
    

    def setupUii(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1366,768)
        BG_Image='I3.jpg'
        image = cv2.imread(BG_Image)
        image=cv2.resize(image, (1366,768))
        BG_Image='III3.jpg'
        cv2.imwrite(BG_Image, image) 
        MainWindow.setStyleSheet(_fromUtf8("\n""background-image: url('III3.jpg');\n"""))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        
        # TITLE
        self.u_user_label2 = QtWidgets.QLabel(MainWindow)
        self.u_user_label2.setGeometry(QtCore.QRect(300, 50, 300, 50))
        self.u_user_label2.setObjectName(_fromUtf8("u_user_label2"))
        self.u_user_label2.setFont(QFont('Times', 10))
        self.u_user_label2.setStyleSheet("background-image: url(milkwhite.jpg);;border: 2px solid magneta")
        entered_text='LUNG CANCER DETECTION'
        self.u_user_label2.setText(f"Project Name: {entered_text}")

        #IMAGE
        self.L = QtWidgets.QLabel(MainWindow)
        self.L.setGeometry(QtCore.QRect(600, 100, 50, 50))
        # Load image
        self.pixmap = QPixmap('logo.png')
        # Set image to label
        self.L.setPixmap(self.pixmap)
        self.L.resize(self.pixmap.width(),self.pixmap.height())


        # EDIT TEXT
        '''
        self.b = QtWidgets.QPlainTextEdit(MainWindow)
        self.b.setGeometry(QtCore.QRect(300, 20, 300, 30))
        self.b.insertPlainText("USER-\n")
        self.b.setStyleSheet("background-image: url(milkwhite.jpg);;border: 2px solid magneta")
        '''

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(250, 300, 131, 27))
        self.pushButton.clicked.connect(self.quit)
        self.pushButton.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n""color: rgb(0, 0, 0);"))
        self.pushButton.setStyleSheet("background-image: url(blue.jpg);;border: 2px solid red")
        self.pushButton.setObjectName(_fromUtf8("pushButton"))

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(250, 180, 131, 27))
        self.pushButton_2.clicked.connect(self.show1)
        self.pushButton_2.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n""color: rgb(0, 0, 0);"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_2.setStyleSheet("background-image: url(white.jpg);;border: 2px solid red")
        
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(250, 220, 131, 27))
        self.pushButton_4.clicked.connect(self.show2)
        self.pushButton_4.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n""color: rgb(0, 0, 0);"))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.pushButton_4.setStyleSheet("background-image: url(white.jpg);;border: 2px solid red")

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(250, 260, 131, 27))
        self.pushButton_5.clicked.connect(self.show3)
        self.pushButton_5.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n""color: rgb(0, 0, 0);"))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        self.pushButton_5.setStyleSheet("background-image: url(white.jpg);;border: 2px solid red")
        
        

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
       
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "lungs cancer DETECTION ", None))
        self.pushButton_2.setText(_translate("MainWindow", "TEST", None))
        self.pushButton_4.setText(_translate("MainWindow", "TRAIN", None))
        self.pushButton_5.setText(_translate("MainWindow", "RESULT", None))
        self.pushButton.setText(_translate("MainWindow", "Exit", None))

    def quit(self):
        print ('Process end')
        print ('******End******')
        app.quit()
        self.close()
        quit()

         
    def show1(self):
        image_path= filedialog.askopenfilename(filetypes = (("BROWSE  IMAGE", "*.png"), ("All files", "*")))
        #root.destroy()
        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        img=cv2.imread(image_path)
        cv2.imshow('INPUT IMAGE',img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()

        # RESIZING
        img = cv2.resize(img,(512,512),3)
        cv2.imshow('RESIZED IMAGE',img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

        # GRAY CONVERSION
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('GRAY IMAGE',img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
        imgg=img


        # GAUSSIAN FILTERING
        img1 = cv2.GaussianBlur(img,(5,5),0)
        cv2.imshow('GAUSSIAN IMAGE',img1)
        cv2.waitKey(100)
        cv2.destroyAllWindows()

        # MEDIAN FILTERED
        img1 = cv2.medianBlur(img,31)
        cv2.imshow('MEDIAN IMAGE',img1)
        cv2.waitKey(100)
        cv2.destroyAllWindows()


        # OTSU's THRESHOLDING
        ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #thresh= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,255,0)
        kernel = np.ones((31,31),np.uint8)
        thresh= cv2.dilate(thresh,kernel,iterations = 1)

        PALM = cv2.bitwise_and(imgg, thresh)
        cv2.imshow('SEGMENTED IMAGE',PALM)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line
            in tf.gfile.FastGFile("retrained_labels.txt")]
        # Unpersists graph from file
        with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        # Feed the image_data as input to the graph and get first prediction
        with tf.Session() as sess:
            # Get predictions
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            score0 = predictions[0][0]  # Confidence score for Cancer class
            score1 = predictions[0][1]  # Confidence score for Normal class
            TT = [score0, score1]

            # Print scores and prediction result
            print(TT)
            rr = np.argmax(TT)  # Find the class with the highest confidence
            print(rr)
            print('-------------------------------------')


        if score1 > 0.97:  # Predicted as Normal
            print('Result \n')
            print('GIVEN IMAGE BELONGS TO NORMAL CLASS \n')
            messagebox.showinfo(title='NORMAL:', message='Similarity Score: ' + str(np.max(TT)))

        elif 0.91 <= score0 < 0.99 and 0.02 <= score1 <= 0.05:  # Squamous Cell Carcinoma
            print('GIVEN IMAGE BELONGS TO squamous CANCER CLASS\n')
            messagebox.showinfo(title='CANCER:', message='SCC Type')

        elif 0.85 <= score0 < 0.99 and 0.003 <= score1 <= 0.13:  # Adenocarcinoma
            print('GIVEN IMAGE BELONGS TO adenocarcinoma CANCER CLASS\n')
            messagebox.showinfo(title='CANCER:', message='Adenocarcinoma Type')

        else:
            print("GIVEN IMAGE BELONGS TO large cell carcinoma CANCER CLASS\n")
            messagebox.showinfo(title='CANCER:', message='Large Cell Carcinoma')

        


    def show2(self):
        s=1
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from keras.layers import Input, Dense
        from  keras import regularizers
        from  keras.models import Sequential, Model
        from  keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
        from  keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
        from keras.layers import Concatenate
        from keras.preprocessing.image import ImageDataGenerator
        from keras.optimizers import Adam, SGD
        import pickle

        # define parameters
        CLASS_NUM = 3
        BATCH_SIZE = 16
        EPOCH_STEPS = int(4323/BATCH_SIZE)
        IMAGE_SHAPE = (512, 512, 3)
        IMAGE_TRAIN = 'retrained_graph'
        MODEL_NAME = 'retrained_graph.h5'
        def inception(x, filters):
                path1 = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                
                path2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(path2)
                
                path3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(path3)
                
                path4 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
                path4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(path4)
                return Concatenate(axis=-1)([path1,path2,path3,path4])
        def auxiliary(x, name=None):
                layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
                layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
                layer = Flatten()(layer)
                layer = Dense(units=256, activation='relu')(layer)
                layer = Dropout(0.4)(layer)
                layer = Dense(units=CLASS_NUM, activation='softmax', name=name)(layer)
                return layer
        def googlenet():
                layer_in = Input(shape=IMAGE_SHAPE)
                layer = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(layer_in)
                layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
                layer = BatchNormalization()(layer)
                layer = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
                layer = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
                layer = BatchNormalization()(layer)
                layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
                layer = inception(layer, [ 64,  (96,128), (16,32), 32]) #3a
                layer = inception(layer, [128, (128,192), (32,96), 64]) #3b
                layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
                layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a
                aux1  = auxiliary(layer, name='aux1')
                layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b
                layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c
                layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d
                aux2  = auxiliary(layer, name='aux2')
                layer = inception(layer, [256, (160,320), (32,128), 128]) #4e
                layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
                layer = inception(layer, [256, (160,320), (32,128), 128]) #5a
                layer = inception(layer, [384, (192,384), (48,128), 128]) #5b
                layer = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(layer)
                layer = Flatten()(layer)
                layer = Dropout(0.4)(layer)
                layer = Dense(units=256, activation='linear')(layer)
                main = Dense(units=CLASS_NUM, activation='softmax', name='main')(layer)
                model = Model(inputs=layer_in, outputs=[main, aux1, aux2])  
                return model

        model= googlenet()
        model.summary()



    def show3(self):
        print('RES')
        file= open("Gnet.h5",'rb')
        cnf_matrix = pickle.load(file)
        file.close()

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix[0:3,0:3], classes=['Normal ','squamous','adenocarcinoma'], normalize=True,title='Proposed Method')
        plt.show()
                


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen()
    print('Screen: %s' % screen.name())
    size = screen.size()
    print('Size: %d x %d' % (size.width(), size.height()))
    MainWindow = QtWidgets.QMainWindow()
    #MainWindow.showFullScreen()
    ui = Ui_MainWindow1()
    ui.setupUii(MainWindow)
    MainWindow.move(0, 0)
    MainWindow.show()
    sys.exit(app.exec_())


