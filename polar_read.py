
from typing import ValuesView
import cv2
import numpy as np
#import paho.mqtt.client as mqtt
import time
import math
import pandas as pd
import itertools
import heapq

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

#计算点到直线的距离
def getDist_P2L(PointP,Pointa,Pointb):
    
    A=0
    B=0
    C=0
    A=Pointa[1]-Pointb[1]
    B=Pointb[0]-Pointa[0]
    C=Pointa[0]*Pointb[1]-Pointa[1]*Pointb[0]
    distance=0
    distance=(A*PointP[0]+B*PointP[1]+C)/math.sqrt(A*A+B*B)
    
    return distance

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


file_type='png'
name ='531'
gauge_number = 531

#img = cv2.imread('./yuan/22.jpg')
img = cv2.imread('./TEST/1.png')
img = cv2.imread('./buxiugang/531.png')

height, width = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 120, np.array([]), 100, 50, int(height*0.30), int(height*0.40))

a, b, c = circles.shape
xc,yc,r = avg_circles(circles, b)

imgw = img.copy()

#draw center and circle
cv2.circle(imgw, (xc, yc), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
cv2.circle(imgw, (xc, yc), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

print("xc,yc,r,",xc,yc,r)

canny= cv2.Canny(gray, 100, 10)
cv2.namedWindow("polar",cv2.WINDOW_NORMAL)
cv2.namedWindow("polar5",cv2.WINDOW_NORMAL)

ro,col,_=img.shape
cent=(int(col/2),int(ro/2))

thresh =120
maxValue = 255
th, dst5 = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY_INV)
canny5= cv2.Canny(dst5, 100, 10)


polar5=cv2.linearPolar(dst5,(xc,yc),r,cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
sums = np.sum(polar5,axis=1)
posi = np.argmax(sums)
polar52 = cv2.cvtColor(polar5,cv2.COLOR_GRAY2BGR)
cv2.line(polar52, ( int(0),int(posi)), (int(col),int(posi)),(255, 0,255 ), 3)
#cv2.imshow('dst5line', polar52)
cv2.imwrite('%s-polar5line-%s.%s' % (name, gauge_number,file_type), polar52)

zhenpos,qipos,zhipos=0,0,0
zhenpos=int(posi)


sumss =[]
for col in range(int(col*0.5),int(col-col*0.08)):
    cc = polar5[:,col]
    tt = [abs(cc[i+1]-cc[i]) for i in range(0,len(cc)-1,2)]
    sumss.append(sum(tt))


posi = np.argmax(sumss)
polar53 = cv2.cvtColor(polar5,cv2.COLOR_GRAY2BGR)
cv2.line(polar53, (int(posi)+int(col*0.5)+10, int(0)), (int(posi)+int(col*0.5)+10,int(ro)),(255, 0,255 ), 3)
cv2.imshow('dst5line3', polar53)
cv2.imwrite('%s-polar5line3-%s.%s' % (name, gauge_number,file_type), polar53)


imgtt= img.copy()


separation= 10 
interval = int(360/separation)
p3 = np.zeros((interval,2)) 
p4 = np.zeros((interval,2))

for i in range(0,interval):
    for j in range(0,2):
        if (j%2==0):
            #33  0.99  11  0.9  22  也是个经验值
            p3[i][j] = xc + 0.9 * r * np.cos(separation * i * np.pi / 180) #point for lines
        else:
            p3[i][j] = yc + 0.9 * r * np.sin(separation * i * np.pi / 180)


def region_of_interest(img, vertices):
    mask= np.zeros_like(img)
    match_mask_color= 255
    cv2.fillPoly(mask, vertices, match_mask_color)   
    #可以更改为 环
    cv2.imwrite('%s-mask-%s.%s' % (name, gauge_number,file_type), mask)
    masked_image= cv2.bitwise_and(img, mask)
    cv2.imwrite('%s-mask2-%s.%s' % (name, gauge_number,file_type), masked_image)
    return masked_image


canny= cv2.Canny(gray, 200, 20)
cv2.imwrite('%s-canny-%s.%s' % (name, gauge_number,file_type), canny)
region_of_interest_vertices= p3
cropped_image= region_of_interest(canny, np.array([region_of_interest_vertices], np.int32))

cv2.imwrite('%s-crop-%s.%s' % (name, gauge_number,file_type), cropped_image)

cv2.namedWindow("contoursjpg",cv2.WINDOW_NORMAL)
contours3= img.copy()

maskpl= np.zeros_like(cropped_image)

contours, heirarchy= cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
int_cnt= []
for cnt in contours:
    area = cv2.contourArea(cnt)
    [x, y, w, h] = cv2.boundingRect(cnt)
    cpd = dist_2_pts(x+w/2,y+h/2,xc, yc)
    #33  4/4  11  3.5/4  22   也是个经验值
    if area<500 and int(cpd) <r*4/4 and int(cpd) > r*2/4:
        cv2.drawContours(contours3, cnt, -1, (255,0,0), 3)
        cv2.drawContours(maskpl, cnt, -1, 255, 3)
        int_cnt.append(cnt) 
        cv2.imshow('contoursjpg', contours3)
        cv2.waitKey(5)


cv2.imwrite('%s-contours3-%s.%s' % (name, gauge_number,file_type), contours3)


cv2.namedWindow("dst6line3",cv2.WINDOW_NORMAL)
cv2.namedWindow("dst7linedushu",cv2.WINDOW_NORMAL)

ro,col,_=img.shape

polar6=cv2.linearPolar(maskpl,(xc,yc),r,cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)  
cv2.imwrite('%s-polar6-%s.%s' % (name, gauge_number,file_type), polar6)


sum6 =[]
sum6tt =[]
for col in range(int(col)):
    cc = polar6[:,col]
    tt = [abs(cc[i+1]-cc[i]) for i in range(0,len(cc)-1,2)]
    sum6tt.append(tt)  
    sum6.append(sum(tt)) 

posi = np.argmax(sum6)  


polar63 = cv2.cvtColor(polar6,cv2.COLOR_GRAY2BGR)
cv2.line(polar63, (int(posi-3), int(0)), (int(posi-3),int(ro)),(255, 0,255 ), 3)
cv2.imshow('dst6line3', polar63)
cv2.imwrite('%s-polar6line3-%s.%s' % (name, gauge_number,file_type), polar63)


kedulist = sum6tt[posi]  

list1,list2,list3 ,possum= [],[],[],0

for k,v in itertools.groupby(kedulist):
    vv = list(v)
    #print(k,len(vv),possum+len(vv),vv)
    list1.append(k)     
    list2.append(len(vv)) 
    list3.append(possum+len(vv))  

    possum+=len(vv)


max_list2_index = map(list2.index,heapq.nlargest(2,list2))
list02 = list(max_list2_index)

#上面的即 终点位置
shang_end = kedulist[list3[np.min(list02)] - list2[np.min(list02)] -1]
shang_end = 2*(list3[np.min(list02)] - list2[np.min(list02)] -1)

#下面的即 起点位置
xia_start  = kedulist[ list3[np.max(list02)]]
xia_start = 2*(list3[np.max(list02)])

#在极坐标变换后，画指针横线，画起止  三条横线
cv2.line(polar52, ( int(0),int(shang_end)), (int(col),int(shang_end)),(255, 0,255 ), 3)
cv2.line(polar52, ( int(0),int(xia_start)), (int(col),int(xia_start)),(255, 0,255 ), 3)

cv2.imshow('dst7line', polar52)
cv2.imwrite('%s-polar7line-%s.%s' % (name, gauge_number,file_type), polar52)


qipos,zhipos=xia_start,shang_end
dushu = (zhenpos - qipos)/(ro-qipos+zhipos)*(1.0-0.0)

print("dushu: ",dushu)

cv2.putText(img, '%f' %(dushu), (x, y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)

cv2.imshow('dst7linedushu', img)
cv2.imwrite('%s-polar7linedushu-%s.%s' % (name, gauge_number,file_type), img)

cv2.waitKey()


