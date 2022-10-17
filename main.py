import cv2
import numpy as np
from skimage.filters import gaussian
from external_energy import external_energy
from internal_energy_matrix import get_matrix
import matplotlib.pyplot as plt
import math
from scipy.interpolate import CubicSpline
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id



def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        xs.append(x)
        ys.append(y)

        #display point
        cv2.circle(img2, (x, y), 3, 128, -1)
        cv2.imshow('image', img2)


if __name__ == '__main__':

    img_path = "images/star.png"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=gaussian(img,0.3)

    print(img.shape)
    # #point initialization
    img2=img.copy()
    xs = []
    ys = []
    cv2.imshow('image', img2)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # #selected points are in xs and ys

    n_=len(xs)
    int_n=1
    intx=[]
    inty=[]
    if int_n>0:
        for i in range(0,n_-1):
            intx.append(xs[i])
            inty.append(ys[i])
            dx=(xs[i+1]-xs[i])/int_n
            dy = (ys[i + 1] - ys[i])/int_n

            for k in range(1,int_n+1):
                intx.append(xs[i]+(k*dx))
                inty.append(ys[i]+(k*dy))
        intx.append(xs[-1] )
        inty.append(ys[-1])
        xs=intx.copy()
        ys=inty.copy()

    ite=1
    alpha =7
    beta = -1.2
    gamma = 70
    kappa = 2.
    num_points = len(xs)
    img=(img-img.max())/(img.max()-img.min())   #get matrix
    M = get_matrix(alpha, beta, gamma, num_points)

    #get external energy
    w_line = 0.5
    w_edge = 0.5
    w_term = 2.5
    E = external_energy(img, w_line, w_edge, w_term)
    gx=cv2.Sobel(E,cv2.CV_64F, dx=1, dy=0)
    gy = cv2.Sobel(E, cv2.CV_64F, dx=0, dy=1)


    xs=np.array(xs)
    ys = np.array(ys)

    i,j=0,0
    x_s=[]
    y_s=[]
    print(len(xs),len(ys))
    fy=bilinear_interpolate(gy,xs,ys)

    fx=bilinear_interpolate(gx,xs,ys)
    x=xs.copy()
    y=ys.copy()

    all_shapes=[]
    iters=0
    print(M.shape)
    flag=True
    while iters<=ite:
        x_s=np.dot(M,(gamma*xs)+(kappa*fx))
        y_s=np.dot(M,(gamma*ys)+(kappa*fy))
        iters=iters+1
        x_s[x_s < 0] = 0.
        y_s[y_s < 0] = 0.
        x_s[x_s > img.shape[1] - 1] = img.shape[1] - 1
        y_s[y_s > img.shape[0] - 1] = img.shape[0] - 1
        xs = x_s.copy()
        ys = y_s.copy()
        all_shapes.append((x_s.copy(), y_s.copy()))
        fy = bilinear_interpolate(gy,xs,ys)
        fx = bilinear_interpolate(gx,xs,ys)


    ci=0
    c2=0

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], c=(0, 1, 0), lw=0.5)
    cnt=0
    for i in range(0,len(all_shapes)-1):
        if cnt%10==0:
            ax.plot(np.r_[all_shapes[i][0], all_shapes[i][0][0]], np.r_[all_shapes[i][1], all_shapes[i][1][0]], c=(0, 0, 1), lw=0.5)
        cnt+=1

    ax.plot(np.r_[all_shapes[-1][0], all_shapes[-1][0][0]], np.r_[all_shapes[-1][1], all_shapes[-1][1][0]], c=(1, 0, 0), lw=2)

    plt.show()

    ci+=1

