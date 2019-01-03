from random import randrange,randint
from tkinter import *
from tkinter.ttk import Style
import matplotlib.pyplot as plt
from collections import Counter
from PIL import ImageTk, Image
import numpy as np
from tkinter.filedialog import askopenfilename

global img
global imgaseli
def exit():
    exit()

def showimg():
    global canvas
    global img
    img = Image.fromarray(img).convert('RGB')
    img = np.array(img)
    print(img.shape)
    height, width,no_channels = img.shape
    canvas.grid(row=5, column=10)
    photo = ImageTk.PhotoImage(image=Image.fromarray(img))
    canvas.itemconfig(canvas.create_image(0,0, image=photo, anchor=NW))
    canvas.config(width=width, height=height)
    root.mainloop()

def open():
    global img
    global canvas
    global imgaseli
    # filename = askopenfilename()
    filename="lena512color.tiff"
    im = Image.open(filename)
    im = im.convert('RGB')
    img = np.array(im)
    imgaseli = img
    showimg()

def reset():
    global img
    global imgaseli
    img=imgaseli
    img = Image.fromarray(img).convert('RGB')
    img = np.array(img)
    print(img.shape)
    height, width, no_channels = img.shape
    canvas.grid(row=5, column=10)
    photo = ImageTk.PhotoImage(image=Image.fromarray(img))
    canvas.itemconfig(canvas.create_image(0, 0, image=photo, anchor=NW))
    canvas.config(width=width, height=height)
    root.mainloop()

def gray():
    global img
    global canvas
    img = 0.6 * img[:, :, 0] + 0.2 * img[:, :, 1] + 0.2 * img[:, :, 2]
    showimg()

def zoomin():
    global img
    global canvas
    [tinggi, lebar, panjang] = img.shape
    tinggi_baru = tinggi * 2
    lebar_baru = lebar * 2
    image_baru = np.uint8(np.zeros((tinggi_baru, lebar_baru, panjang)))
    for i in range(0, tinggi) :
        for j in range(0, lebar) :
            temp1 = img[i, j, 0]
            temp2 = img[i, j, 1]
            temp3 = img[i, j, 2]
            for k in range(0, 2):
                for l in range(0, 2):
                    dummyT = ((i * 2) + k)
                    dummyL = ((j * 2) + l)
                    image_baru[dummyT, dummyL, 0] = temp1
                    image_baru[dummyT, dummyL, 1] = temp2
                    image_baru[dummyT, dummyL, 2] = temp3
    print(img)
    print(image_baru[:,:,0])
    img=image_baru
    showimg()

def zoomout():
    global img
    [tinggi, lebar, panjang] = img.shape
    tinggi_baru = int(round(tinggi / 2))
    lebar_baru = int(round(lebar / 2))
    image_baru = np.uint8(np.zeros((tinggi_baru, lebar_baru, panjang)))
    i = 0
    j = 0
    for k in range(0,tinggi,2) :
        for l in range(0,lebar,2):
            image_baru[i, j,:]=img[k, l,:]
            j = j + 1
        i = i + 1
        j = 0
    img=image_baru
    showimg()

def cropatas():
    global img
    [tinggi, lebar, panjang] = img.shape
    tinggi_baru = tinggi-2
    image_baru = np.uint8(np.zeros((tinggi_baru, lebar, panjang)))
    for k in range(0,tinggi_baru) :
        image_baru[k, :,:]=img[k+2, :,:]
    img=image_baru
    showimg()

def cropbawah():
    global img
    [tinggi, lebar, panjang] = img.shape
    tinggi_baru = tinggi-2
    image_baru = np.uint8(np.zeros((tinggi_baru, lebar, panjang)))
    for k in range(tinggi_baru-1,-1,-1) :
        image_baru[k, :,:]=img[k, :,:]
    img=image_baru
    showimg()

def cropkiri():
    global img
    [tinggi, lebar, panjang] = img.shape
    lebar_baru = lebar-2
    image_baru = np.uint8(np.zeros((tinggi, lebar_baru, panjang)))
    for k in range(1,lebar_baru) :
        image_baru[:, k,:]=img[:, k+2,:]
    img=image_baru
    showimg()

def cropkanan():
    global img
    [tinggi, lebar, panjang] = img.shape
    lebar_baru = lebar - 2
    image_baru = np.uint8(np.zeros((tinggi, lebar_baru, panjang)))
    for k in range(lebar_baru - 1, -1, -1):
        image_baru[:, k, :] = img[:, k, :]
    img = image_baru
    # print(img[ - 5,  - 5,:])
    showimg()

def brightness():
    global img
    brightness = 50
    contrast = 30
    img=np.int16(img)
    img = img * 50#(contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    showimg()

def brightnessout():
    global img
    brightness = -50
    contrast = -30
    img=np.int16(img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    showimg()

def histogram():
    global img
    list_histogramR = np.zeros((257, 1))
    list_histogramG = np.zeros((257, 1))
    list_histogramB = np.zeros((257, 1))
    [tinggi, lebar, panjang] = img.shape
    print(tinggi,lebar,panjang, list_histogramR.shape, img.shape)
    for j in range(0, tinggi-1):

        for k in range(0, lebar-1):
            # print(img[j, k, 0] + 1)
            list_histogramR[img[j, k, 0] + 1] = list_histogramR[img[j, k, 0] + 1] + 1
            list_histogramG[img[j, k, 1] + 1] = list_histogramG[img[j, k, 1] + 1] + 1
            list_histogramB[img[j, k, 2] + 1] = list_histogramB[img[j, k, 2] + 1] + 1

    Horis =np.array([0, 257])
    N = len(list_histogramR)
    x = range(N)
    plt.hist(list_histogramR)
    plt.show()
    plt.hist(list_histogramG)
    plt.show()
    plt.hist(list_histogramB)
    plt.show()

def blur():
    global img
    [tinggi, lebar, panjang] = img.shape
    image_baru = img.astype(float)
    y = np.array([[1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9]])
    for i in range(0, tinggi - 1):
        for j in range(0, lebar - 1):
            jum = image_baru[i - 1, j - 1,:]*y[0, 0]+image_baru[i, j - 1,:]*y[1, 0]+image_baru[i + 1, j - 1,:]*y[2, 0]+\
                  image_baru[i - 1, j,:]*y[0, 1]+ image_baru[i, j,:]*y[1, 1]+image_baru[i + 1, j,:]*y[2, 1]+\
                  image_baru[i - 1, j + 1,:]*y[0, 2]+image_baru[i, j + 1,:]*y[1, 2]+image_baru[i + 1, j + 1,:]*y[2, 2]\

            img[i , j ,:]=jum
    showimg()

def sharp():
    global img
    [tinggi, lebar, panjang] = img.shape
    image_baru = img.astype(float)
    y = np.array([[0, -1,  0],
                  [-1, 5, -1],
                  [0, -1,  0]])
    for i in range(0, tinggi - 1):
        for j in range(0, lebar - 1):
            jum = image_baru[i - 1, j - 1,:]*y[0, 0]+image_baru[i, j - 1,:]*y[1, 0]+\
                  image_baru[i + 1, j - 1,:]*y[2, 0]+image_baru[i - 1, j,:]*y[0, 1]+ \
                  image_baru[i, j,:]*y[1, 1]+image_baru[i + 1, j,:]*y[2, 1]+\
                  image_baru[i - 1, j + 1,:]*y[0, 2]+image_baru[i, j + 1,:]*y[1, 2]+\
                  image_baru[i + 1, j + 1,:]*y[2, 2]
            img[i , j ,:]=jum
    showimg()

def edgeskeleton():
    global img
    img = 0.6 * img[:, :, 0] + 0.2 * img[:, :, 1] + 0.2 * img[:, :, 2]
    [tinggi, lebar] = img.shape
    image_baru = img.astype(float)
    y = np.array([[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]])
    for i in range(0, tinggi - 1):
        for j in range(0, lebar - 1):
            jum = image_baru[i - 1, j - 1] * y[0, 0] + image_baru[i, j - 1] * y[1, 0] + \
                  image_baru[i + 1, j - 1] * y[2, 0] + image_baru[i - 1, j] * y[0, 1] + \
                  image_baru[i, j] * y[1, 1] + image_baru[i + 1, j] * y[2, 1] + \
                  image_baru[i - 1, j + 1] * y[0, 2] + image_baru[i, j + 1] * y[1, 2] + \
                  image_baru[i + 1, j + 1] * y[2, 2]
            img[i , j ] = jum
    showimg()

def threshold_segmentation():
    global img
    img = 0.6 * img[:, :, 0] + 0.2 * img[:, :, 1] + 0.2 * img[:, :, 2]
    [tinggi, lebar] = img.shape
    image_baru = np.zeros_like(img)

    for i in range(tinggi):
        for j in range(lebar):
            if img[i, j] >= 200 :
                image_baru[i, j] = 255
    img = image_baru
    showimg()

def region_growth():
    global img
    img = 0.6 * img[:, :, 0] + 0.2 * img[:, :, 1] + 0.2 * img[:, :, 2]
    image_baru = np.zeros_like(img)
    h, w = img.shape
    visit = np.zeros_like(img)
    # for i in range(1, tinggi-1):
    #     for j in range(1, lebar-1):
    #         if (img[i, j] - seed <= img[i-1, j-1]) :#and (img_arr[i, j, k] + seed >= img_arr[i-1, j-1, k])):
    #             image_baru[i-1, j-1] = img[i, j]




    # seedX = randint(1, w-2)
    # seedY = randint(1, h-2)
    seedX = 300
    seedY = 250
    # pixels = [[Pixel(x, y, pxlValueMap[x, y]) for x in range(w)] for y in range(h)]

    stillSearching = True

    while stillSearching:
        x = []
        y = []

        # print(seedX, seedY)
        visit[seedX,seedY] = 1
        # image_baru[seedX, seedX] = 255
        if (img[seedX,seedY]>=150):
            image_baru[seedX , seedX] =255
        if(visit[seedX,seedY]==0):
            x.append(seedX)
            y.append(seedY)

        # if (img[seedX,seedY-1]>=150):
        #     img[seedX, seedY - 1] =255
        if (visit[seedX, seedY - 1] ==0):
            x.append(seedX)
            y.append(seedY-1)
        #
        # if (img[seedX,seedY+1]>=150):
        #     img[seedX, seedY + 1] =255
        if (visit[seedX, seedY + 1] ==0):
            x.append(seedX)
            y.append(seedY+1)
        #
        # if (img[seedX-1,seedY]>=150):
        #     img[seedX - 1, seedY] =255
        if (visit[seedX - 1, seedY] ==0):
            x.append(seedX-1)
            y.append(seedY)
        #
        # if (img[seedX-1,seedY-1]>=150):
        #     img[seedX - 1, seedY - 1] =255
        if (visit[seedX - 1, seedY - 1] ==0):
            x.append(seedX-1)
            y.append(seedY-1)
        #
        # if (img[seedX-1,seedY+1]>=150):
        #     img[seedX - 1, seedY + 1] =255
        if (visit[seedX - 1, seedY + 1]==0):
            x.append(seedX - 1)
            y.append(seedY + 1)
        #
        # if (img[seedX+1,seedY]>=150):
        #     img[seedX + 1, seedY] =255
        if (visit[seedX + 1, seedY] ==0):
            x.append(seedX+1)
            y.append(seedY)
        #
        # if (img[seedX+1,seedY-1]>=150):
        #     img[seedX + 1, seedY - 1] =255
        if (visit[seedX + 1, seedY - 1] ==0):
            x.append(seedX+1)
            y.append(seedY-1)
        #
        # if (img[seedX+1,seedY+1]>=150):
        #     img[seedX + 1, seedY + 1] =255
        if (visit[seedX + 1, seedY + 1] ==0):
            x.append(seedX+1)
            y.append(seedY+1)
        print( x)
        if(len(x)==0):
            stillSearching = False
        else:
            random_index = randrange(len(x))
            seedX = x[random_index]
            seedY = y[random_index]
            # print(x[random_index], y[random_index])

    img = image_baru
    showimg()

root = Tk()
root.title("GUI")
Style().configure("TButton", padding=(0, 5, 0, 5),
    font='serif 10')
btnopen = Button(master=root, text="show image", command=open)
btnopen.grid(row=0, column=0)
btngray = Button(master=root, text="Gray", command=gray)
btngray.grid(row=0, column=1)
btnzoomin = Button(master=root, text="Zoom", command=zoomin)
btnzoomin.grid(row=0, column=2)
# btnzoomin.pack()
btnzoomout = Button(master=root, text="Zoomout", command=zoomout)
btnzoomout.grid(row=0, column=3)
# btnzoomout.pack()
btncropatas = Button(master=root, text="Crop Atas", command=cropatas)
btncropatas.grid(row=0, column=5)
btncropbawah = Button(master=root, text="Crop Bawah", command=cropbawah)
btncropbawah.grid(row=1, column=5)
btncropkanan = Button(master=root, text="Crop Kanan", command=cropkanan)
btncropkanan.grid(row=1, column=6)
btncropkiri = Button(master=root, text="Crop Kiri", command=cropkiri)
btncropkiri.grid(row=1, column=4)
btnbrightness = Button(master=root, text="Brightness", command=brightness)
btnbrightness.grid(row=1, column=0)
btnbrightnessout = Button(master=root, text="Brightness Out", command=brightnessout)
btnbrightnessout.grid(row=1, column=1)
btnhistogram = Button(master=root, text="Histogram", command=histogram)
btnhistogram.grid(row=2, column=0)
btnblur = Button(master=root, text="Blur", command=blur)
btnblur.grid(row=3, column=0)
btnsharp = Button(master=root, text="Sharp", command=sharp)
btnsharp.grid(row=3, column=1)
btnedge = Button(master=root, text="Edge Skeleton", command=edgeskeleton)
btnedge.grid(row=3, column=2)
btnthre = Button(master=root, text="Threshold Segmentasion", command=threshold_segmentation)
btnthre.grid(row=4, column=0)
btnseed = Button(master=root, text="Region Growth", command=region_growth)
btnseed.grid(row=4, column=1)
btnreset = Button(master=root, text="Reset", command=reset)
btnreset.grid(row=5, column=0)
# btnexit = Button(master=root, text="exit", command=exit)
# btnexit.grid(row=6, column=0)
# btnexit.pack()
canvas = Canvas(root)
# canvas = Canvas(root)



root.mainloop()