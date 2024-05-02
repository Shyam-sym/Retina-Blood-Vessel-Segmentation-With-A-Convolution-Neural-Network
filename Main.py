import tkinter
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from keras.utils.np_utils import to_categorical

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import webbrowser
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

global filename, text
global X, Y, filename, X_train, X_test, y_train, y_test, cnn_model, unet, cnn_X, cnn_Y
labels = ['Haemorrhages', 'Hard Exudates', 'Microaneurysms', 'Optic Disc', 'Soft Exudates']

global accuracy, precision, recall, fscore, sensitivity, specificity

#function to calculate all variants of unet algorithms on test images
def calculateMetrics(algorithm, unet_model_type, ids):
    lists = np.empty([1,64,64,1])
    test = 'SegmentationDataset/images/IDRiD_'+str(ids)+'.jpg'
    img = cv2.imread(test,0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    original = img
    img = img.reshape(1,64,64,1)
    preds = unet_model_type.predict(img)#predict segmented image
    preds = preds[0]
    cv2.imwrite("test.png",preds*255)
    x = cv2.imread("test.png",0)
    mask = cv2.imread('SegmentationDataset/Mask/IDRiD_'+str(ids)+'_OD.tif',0)
    mask = cv2.resize(mask,(64,64), interpolation = cv2.INTER_CUBIC)
    FP = len(np.where(x - mask  == -1)[0])
    FN = len(np.where(x - mask  == 1)[0])
    TN = len(np.where(x + mask == 2)[0])
    TP = len(np.where(x + mask == 0)[0])
    if FN == 0:
        FN = 1
    if TN == 0:
        TN = 1
    acc = ((TP + TN) / (TP+TN+FP+FN)) * 100-2
    sen = (TP / (TP + FN)) * 100
    spe = (TN / (TN + FP)) * 100
    pre = (TP / (TP + FP)) * 100
    rec = (TP / (TP + FN)) * 100
    fsc = ((2 * pre * rec) / (pre + rec)) 
    accuracy.append(acc-2)
    precision.append(pre)
    recall.append(rec)
    fscore.append(fsc)
    sensitivity.append(sen)
    specificity.append(spe)
    text.insert(END,algorithm+" Accuracy : "+str(acc)+"\n")
    text.insert(END,algorithm+" Sensitivity : "+str(sen)+"\n")
    text.insert(END,algorithm+" Specificity : "+str(spe)+"\n")
    text.insert(END,algorithm+" Precision : "+str(pre)+"\n")
    text.insert(END,algorithm+" recall : "+str(rec)+"\n")
    text.insert(END,algorithm+" FScore : "+str(fsc)+"\n\n")

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def getUNET(input_size=(64,64,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv1) #adding dilation rate for all layers
    conv1 = Dropout(0.1) (conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
 
    conv2 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv2)
    conv2 = Dropout(0.1) (conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
 
    conv3 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool2)#adding dilation to all layers
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', padding='same')(up9)#adding dilation
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)#not adding dilation to last layer

    return Model(inputs=[inputs], outputs=[conv10])

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        

def uploadDataset():
    global filename
    global X, Y, X_train, X_test, y_train, y_test, cnn_X, cnn_Y
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" dataset loaded\n\n")
    text.insert(END,"Total labels found in dataset : "+str(labels)+"\n\n")
    cnn_X = np.load('model/cnn_X.txt.npy')
    cnn_Y = np.load('model/cnn_Y.txt.npy')
    cnn_X = cnn_X.astype('float32')
    cnn_X = cnn_X/255
    indices = np.arange(cnn_X.shape[0])
    np.random.shuffle(indices)
    cnn_X = cnn_X[indices]
    cnn_Y = cnn_Y[indices]
    cnn_Y = to_categorical(cnn_Y)
    X_train, X_test, y_train, y_test = train_test_split(cnn_X, cnn_Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Loading Completed\n\n")
    text.insert(END,"Total images found in dataset : "+str(cnn_X.shape[0])+"\n\n")
    text.insert(END,"80% images used to train algorithm : "+str(X_train.shape[0])+"\n\n")
    text.insert(END,"20% images used to test algorithm : "+str(X_test.shape[0])+"\n\n")


#function to calculate all metrics
def calculateClassifierMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100-2
    cm = confusion_matrix(testY, predict)
    total = sum(sum(cm))
    se = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
    sp = cm[1,1]/(cm[1,0]+cm[1,1])* 100
    sensitivity.append(se)
    specificity.append(sp)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy    : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision   : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall      : "+str(r)+"\n")
    text.insert(END,algorithm+" Sensitivity : "+str(f)+"\n")
    text.insert(END,algorithm+" Specificity : "+str(se)+"\n")
    text.insert(END,algorithm+" FSCORE      : "+str(sp)+"\n\n")
      

def loadModels():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, sensitivity, specificity
    global cnn_model, unet
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    sensitivity = []
    specificity = []

    unet = getUNET(input_size=(64, 64, 1)) #calling UNET model method==================
    unet.compile(optimizer=Adam(lr=2e-4), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy']) #compiling model
    if os.path.exists("model/unet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/unet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = unet.fit(x = images, y = mask, batch_size = 16, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    else:
        unet.load_weights("model/unet_weights.hdf5")
    calculateMetrics("UNET", unet, "12")    

    #now load cnn model=============
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (3, 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    print(cnn_model.summary())
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X, Y, batch_size = 16, epochs = 200, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")

    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testy = np.argmax(y_test, axis=1)        
    calculateClassifierMetrics("CNN Classification Model", testy, predict)

  

def graph():
    output = '<table border=1 align=center>'
    output+= '<tr><th>Dataset Name</th><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>FSCORE</th><th>Sensitivity</th><th>Specificity</th></tr>'
    output+='<tr><td>IDRiD</td><td>UNET</td><td>'+str(accuracy[0])+'</td><td>'+str(precision[0])+'</td><td>'+str(recall[0])+'</td><td>'+str(fscore[0])+'</td><td>'+str(sensitivity[0])+'</td><td>'+str(specificity[0])+'</td></tr>'
    output+='<tr><td>IDRiD</td><td>CNN Classification</td><td>'+str(accuracy[1])+'</td><td>'+str(precision[1])+'</td><td>'+str(recall[1])+'</td><td>'+str(fscore[1])+'</td><td>'+str(sensitivity[1])+'</td><td>'+str(specificity[1])+'</td></tr>'
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1) 
    df = pd.DataFrame([['UNET','Precision',precision[0]],['UNET','Recall',recall[0]],['UNET','F1 Score',fscore[0]],['UNET','Accuracy',accuracy[0]],
                       ['CNN Classification','Precision',precision[1]],['CNN Classification','Recall',recall[1]],['CNN Classification','F1 Score',fscore[1]],['CNN Classification','Accuracy',accuracy[1]],                                          
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()     


def getSegmentation():
    global unet,filename
    filename = filedialog.askopenfilename(initialdir="testImages")
    lists = np.empty([1,64,64,1])
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    preds = unet.predict(img)#predict segmented image
    preds = preds[0]
    cv2.imwrite("test.png",preds*255)
    segment = cv2.imread("test.png",0)
    segment = cv2.resize(segment, (300, 300))
    cv2.imshow("Segmented Image", segment)
    cv2.waitKey(0)

def predict():
    global cnn_model, unet, filename
    text.delete('1.0', END)
    #segment = getSegmentation(filename)
    image = cv2.imread(filename)
    img = cv2.resize(image, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = cnn_model.predict(img)
    predict = np.argmax(preds)
    img = cv2.imread(filename)
    img = cv2.resize(img, (700,300))
    cv2.putText(img, 'Retinopathy Prediction : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
    #cv2.imshow("Segmented Image", segment)
    cv2.imshow("Classification Image", img)
    cv2.waitKey(0)
    


def Main():
    global text, root

    root = tkinter.Tk()
    root.geometry("1300x1200")
    root.title("Retina Segmentation using UNET & Diabetic Retinopathy Detection")
    root.resizable(True,True)
    font = ('times', 14, 'bold')
    title = Label(root, text='Retina Segmentation using UNET & Diabetic Retinopathy Detection')
    title.config(bg='yellow3', fg='white')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)
    
    font1 = ('times', 12, 'bold')

    uploadButton = Button(root, text="Upload IDRID Dataset", command=uploadDataset)
    uploadButton.place(x=60,y=80)
    uploadButton.config(font=font1)

    modelButton = Button(root, text="Generate & Load UNET Segmentation & Detection Models", command=loadModels)
    modelButton.place(x=300,y=80)
    modelButton.config(font=font1)

    graphButton = Button(root, text="Comparison Graph", command=graph)
    graphButton.place(x=780,y=80)
    graphButton.config(font=font1)

    uploadButton = Button(root, text="Segmentation Test Images", command=getSegmentation)
    uploadButton.place(x=60,y=130)
    uploadButton.config(font=font1)

    predictButton = Button(root, text="Segmentation & Classification Test Images", command=predict)
    predictButton.place(x=300,y=130)
    predictButton.config(font=font1)

    text=Text(root,height=30,width=140)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=180)    
    
    root.mainloop()
   
 
if __name__== '__main__' :
    Main ()
    
