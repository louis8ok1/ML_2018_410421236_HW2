import numpy as np  
import pandas as pd  
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding  
from keras.datasets import mnist  #下載 mnist 資料 
import matplotlib.pyplot as plt
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import Dropout
np.random.seed(10)  

#########
#下載 Mnist 資料
########
#==================================================================#
#讀取與查看 mnist 資料 

(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()  
print("train data = {:7,}".format(len(X_train_image)))  
print("test  data = {:7,}".format(len(X_test_image)))  
#==================================================================#

#########
#查看訓練資料 
########
#==================================================================#
#看載入資料的長相與格式
#查看單一圖片
def plot_image(image):
    fig=plt.gcf()#通過plt.gcf即可得到當前FIgure的引用
    fig.set_size_inches(2,2)  
    plt.imshow(image, cmap='binary') # cmap='binary' 參數設定以黑白灰階顯示.  
    plt.show()  
plot_image(X_train_image[0])
print("This picture's label is",y_train_label[0])
#==================================================================#
#==================================================================#
#看載入資料的長相與格式
#查看多張圖片
def plot_images_labels_predict(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25:num = 25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title = "l="+str(labels[idx])
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
    plt.show()  
plot_images_labels_predict(X_train_image, y_train_label, [], 0, 10)
print("\n\n")
#==================================================================#
#########
#資料預處理(才能在之後的訓練餵進去)
########
#==================================================================#

x_Train = X_train_image.reshape(60000, 28*28).astype('float32')  
x_Test = X_test_image.reshape(10000, 28*28).astype('float32')  
print(" xTrain: %s" % (str(x_Train.shape)))  
print(" xTest: %s" % (str(x_Test.shape)))  
  
# Normalization  
x_Train_norm = x_Train/255  
x_Test_norm = x_Test/255  
#one-hot encoding
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

#==================================================================#
#########
#建立模型
########
#==================================================================#

model = Sequential()  # Build Linear Model

model.add(Dense(units=1000, input_dim=784, kernel_initializer='normal', activation='relu')) # Add Input/hidden layer  
model.add(Dropout(0.5))#avoid overfitting
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) # Add Input/hidden layer  
model.add(Dropout(0.5))#avoid overfitting
model.add(Dense(units=128, input_dim=784, kernel_initializer='normal', activation='relu')) # Add Input/hidden layer  
model.add(Dropout(0.5))#avoid overfitting

model.add(Dense(units=10, kernel_initializer='normal', activation='softmax')) # Add Hidden/output layer  

print(" Model summary:")  
model.summary()  
#==================================================================#
#########
#進行訓練
########
#==================================================================#
#對訓練模型進行設定
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
#==================================================================#
#開始訓練 
train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2) 
#==================================================================#
#以圖顯示訓練過程 
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  
show_train_history(train_history, 'acc', 'val_acc')  
show_train_history(train_history, 'loss', 'val_loss')  
#==================================================================#
#評估模型準確率 
scores = model.evaluate(x_Test_norm, y_TestOneHot)  
print("")  
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))
#=====================================================================#
#回測
prediction = model.predict_classes(x_Test_norm)
print("\t[Info] Show 10 prediction result (From 240):")  
print("%s\n" % (prediction[100:110]))  

plot_images_labels_predict(X_test_image, y_test_label, prediction, idx=240)  
  

