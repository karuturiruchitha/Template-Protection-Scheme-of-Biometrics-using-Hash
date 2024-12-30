from tkinter import *
import tkinter as tk
import cv2
import os

import pickle
from numpy import save
from keras.utils import np_utils
import os
from tkinter import filedialog
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
#from multimodal_face_and_fp_project import extract_featt
from skimage.color import rgb2gray
import pickle
from tkinter import messagebox
from PIL import ImageTk, Image


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
#from keras.layers import Flatten
from keras.layers import merge, Input
import h5py
from tensorflow.keras.layers import Dense, Activation, Flatten


from tensorflow.keras.models import load_model
model1 = load_model('face_CNN.h5')
model2= load_model('fingerprint_CNN.h5')
model3= load_model('vein_CNN.h5')

import matplotlib.pyplot as plt
from sk_dsp_comm.fec_conv import FECConv
from sk_dsp_comm import digitalcom as dc
import numpy as np
cc = FECConv()





image_input = Input(shape=(160,160,3))
weight_path = "Multi"
model = VGG19(weights="vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False)

from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm

    
# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    print(img_path)
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)


def extract_featt(img):
    #img_path= "D:\\Multi\\dataset\\Face 1\\1.png"
    #img = image.load_img(img_path,target_size=(160, 160))
    img_data = image.img_to_array(img)
    img_data =  np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    flat_feat = features.flatten()
    print(flat_feat.shape)
    return flat_feat

def load_face():
    filename = filedialog.askopenfilename(title='open')
    main_img = cv2.imread(filename)
    f_image= cv2.imread(filename)
    f_image=cv2.resize(image,(250,250))
    return f_image
def preprocess(f_image):
    image=cv2.resize(f_image,(250,250))
    cv2.imshow('Original Image',image)
    mean, std = image.mean(), image.std()
    image = (image-mean)/std
    cv2.imshow('Normalized Image',image)
    cv2.imwrite('Normalized.jpg', image)
    return image

def face_features(image):
    x=extract_featt(image)
    z=x
    #messagebox.showinfo('Feature Extractted ',z)
    return z

def load_fp():
    filename = filedialog.askopenfilename(title='open')
    main_img = cv2.imread(filename)
    fp_image= cv2.imread(filename)
    fp_image=cv2.resize(image,(250,250))
    return fp_image


def preprocess(fp_image):
    image=cv2.resize(fp_image,(250,250))
    cv2.imshow('Original Image',image)
    out = fingerprint_enhancer.enhance_Fingerprint(image)
    cv2.imshow('Enhanced Image',out)
    cv2.imwrite("Enhanced.jpg", out)
    return out

def fp_features(out):
    x=extract_featt(out)
    z1=x
    #messagebox.showinfo('Feature Extractted ',z)
    return z1

def resiz(main_img):
    re_face = cv2.resize(main_img,(160,160))
    mean, std = re_face.mean(), re_face.std()
    re_face = (re_face-mean)/std
    re_face = re_face*225
    #cv2.imshow("kgkjv",re_face)
    return re_face


face_f =[]
fin_f=[]
class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master

        # changing the title of our master widget      
        self.master.title("Multimodal Biometrics Using Deep Hashing ")
        
        self.pack(fill=BOTH, expand=1)
        w = tk.Label(root, 
		 text="Multimodal Biometrics Using Deep Hashing ",
		 fg = "white",
		 bg = "black",
		 font = "Helvetica 20 bold italic")
        w.pack()
        w.place(x=350, y=0)
        #creating buttons
        quitButton = Button(self,command=self.query, text="LOAD FACE IMAGE",fg="black",activebackground="light grey",width=20)
        quitButton.place(x=50, y=100)
        quitButton = Button(self,command=self.query1, text="LOAD FINGERPRINT IMAGE",fg="black",activebackground="light gray",width=20)
        quitButton.place(x=50, y=150)
        quitButton = Button(self,command=self.query2,text="LOAD FINGERVEIN IMAGE",fg="black",activebackground="light grey",width=20)
        quitButton.place(x=50, y=200)
        quitButton = Button(self,command=self.feature,text="FEATURE EXTRACTION",fg="black",activebackground="light grey",width=20)
        quitButton.place(x=50, y=250)
        quitButton = Button(self,command=self.fusion,text="FEATURE FUSION",activebackground="light grey",fg="black",width=20)
        quitButton.place(x=50, y=300)
        quitButton = Button(self,command=self.predict,text="PREDICTION",activebackground="light grey",fg="black",width=20)
        quitButton.place(x=50, y=350)
        
        



        load = Image.open("gray.bmp")
        render = ImageTk.PhotoImage(load)

        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=200, width=200, bg='white')
        image2.image = render
        image2.place(x=250, y=150)

        image3=Label(self, image=render,borderwidth=15, highlightthickness=5, height=200, width=200, bg='white')
        image3.image = render
        image3.place(x=750, y=150)

        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=200, width=200, bg='white')
        image4.image = render
        image4.place(x=500, y=150)



        

#for face_image_load
    def query(self, event=None):
        contents ="Loading Image..."
        global T,rep
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=1000, y=150)
        T.insert(END,contents)
        print(contents)
        rep = filedialog.askopenfilenames() 
        img = cv2.imread(rep[0])
        #cv2.imshow('fff2',img)
        img = cv2.resize(img,(250,250))
        #cv2.imshow('fff1',img)
        Input_img=img.copy()
        print(rep[0])
        #cv2.imshow('fff',Input_img)
        self.from_array = Image.fromarray(cv2.resize(img,(250,250)))
        load = Image.open(rep[0])
        render = ImageTk.PhotoImage(load.resize((250,250)))
        #cv2.imshow('fff',render)
        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=200, width=200, bg='white')
        image2.image = render
        image2.place(x=250, y=150)
        contents="Image Loadeded successfully !!"
        
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=1000, y=150)
        T.insert(END,contents)
        print(contents)
        self.Input_img=Input_img

    def close_window(): 
        Window.destroy()
    

    def feature(self, event=None):
        contents ="Feature Extracting..."
        global T,rep,rep1,rep2

        main_img = cv2.imread(rep[0])
        res=resiz(main_img)
        #re_face = cv2.resize(main_img,(160,160))
        res1=extract_featt(res)
        #messagebox.showinfo('Feature Extractted ',z)
        
        main_img = cv2.imread(rep1[0])
        #re_face = cv2.resize(main_img,(160,160))
        res= cv2.resize(main_img,(160,160))
        res2=extract_featt(res)
        #messagebox.showinfo('Feature Extractted ',z)

        
        main_img = cv2.imread(rep2[0])

        #re_face = cv2.resize(main_img,(160,160))
        res= cv2.resize(main_img,(160,160))
        res3=extract_featt(res)
        #messagebox.showinfo('Feature Extractted ',z)
        X = np.concatenate((res1,res2,res3))
        import hashlib  
        out=X
        out1=np.zeros((out.shape[0]),)
        for i in range(out.shape[0]):
            if out[i]>150:
                out1[i]=1
        z = cc.viterbi_decoder(out1)
        z=str(z)
        result = hashlib.sha256(z.encode())
        final_fea=result.hexdigest()
        contents=final_fea
        np.save('temp_hashing.npy',final_fea)
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=1000, y=150)
        T.insert(END,contents)
        print(contents)
    def query1(self, event=None):
        contents ="Loading Image..."
        global T,rep1
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=1000, y=150)
        T.insert(END,contents)
        print(contents)
        rep1 = filedialog.askopenfilenames() 
        img = cv2.imread(rep1[0])
        #cv2.imshow('fff2',img)
        img = cv2.resize(img,(256,256))
        #cv2.imshow('fff1',img)
        Input_img=img.copy()
        print(rep1[0])
        cv2.imwrite('fin.png',img)
        render = Image.open('fin.png')
        render = ImageTk.PhotoImage(render.resize((250,250)))
        #cv2.imshow('fff',render)
        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=200, width=200, bg='white')
        image4.image = render
        image4.place(x=750, y=150)
        contents="Image Loadeded successfully !!"
        
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=1000, y=150)
        T.insert(END,contents)
        print(contents)
        self.Input_img=Input_img
    def query2(self, event=None):
        contents ="Loading Image..."
        global T,rep2
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=1000, y=150)
        T.insert(END,contents)
        print(contents)
        rep2 = filedialog.askopenfilenames() 
        img = cv2.imread(rep2[0])
        #cv2.imshow('fff2',img)
        img = cv2.resize(img,(256,256))
        #cv2.imshow('fff1',img)
        Input_img=img.copy()
        print(rep2[0])
        cv2.imwrite('fin.png',img)
        render = Image.open('fin.png')
        render = ImageTk.PhotoImage(render.resize((250,250)))
        #cv2.imshow('fff',render)
        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=200, width=200, bg='white')
        image4.image = render
        image4.place(x=500, y=150)
        contents="Image Loadeded successfully !!"
        
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=1000, y=150)
        T.insert(END,contents)
        print(contents)
        self.Input_img=Input_img
    def close_window(): 
        Window.destroy()



    def fusion(self):
        global data
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=1000, y=150)
        T.insert(END,"Fusion Completed ..")
        print(contents)
        
    def predict(self):
        global data,rep,rep1,rep2
        key=np.load('hashing.npy')
        temp_key=np.load('temp_hashing.npy')
        test_tensors = paths_to_tensor(rep[0])/255
        pred1=model1.predict(test_tensors)
        print(np.argmax(pred1))
        print(pred1)

        test_tensors = paths_to_tensor(rep1[0])/255
        pred=model2.predict(test_tensors)
        print(np.argmax(pred))


        test_tensors = paths_to_tensor(rep2[0])/255
        pred=model3.predict(test_tensors)
        print(np.argmax(pred))
        #print(key[0,:],temp_key)
        if np.max(pred1)>.9 :
            contents='Biometric accessed successfully '
            messagebox.showinfo('Biometric accessed successfully ')
        else:
            contents='Access denied '
            messagebox.showinfo('Access denied ')
        
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=1000, y=150)
        T.insert(END,contents)
        print(contents)
        

##data = np.concatenate((face_f,fin_f),axis=0)
##print(data)

root = Tk()
root.geometry("1400x800")
app = Window(root)
root.mainloop()
