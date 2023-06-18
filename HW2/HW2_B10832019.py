'''作業二：OpenCV基本應用 提示'''
#載入相關套件模組
import numpy as np
import cv2 as cv

buttonDown= False #滑鼠左鍵是否按下(全域面數)

## 自定義滑鼠回應函式
def onMouse(event, x, y, flags, param):
	global buttonDown
	#如果滑鼠左鍵按下
	if event == cv.EVENT_LBUTTONDOWN:	
		#buttonDown設為 True
		buttonDown = True
	#如果滑鼠移動
	elif event == cv.EVENT_MOUSEMOVE:
		#如果按鈕按下
		if buttonDown == True:
			#在(x,y)位置繪製半徑6px的黃色圓形點
			cv.circle(im1, (x,y), 6, (0,255,255), -1)
            #顯示影像
			cv.imshow("draw", im1)
	#如果滑鼠左鍵彈起
	elif event == cv.EVENT_LBUTTONUP:
		#buttonDown設為 False
		buttonDown = False
	#如果按下滑鼠右鍵
	elif event == cv.EVENT_RBUTTONDOWN:
		#將視窗刪除
		cv.destroyWindow("draw")


  
#    event: EVENT_LBUTTONDOWN,   EVENT_RBUTTONDOWN,   EVENT_MBUTTONDOWN,
#         EVENT_LBUTTONUP,     EVENT_RBUTTONUP,     EVENT_MBUTTONUP,
#         EVENT_LBUTTONDBLCLK, EVENT_RBUTTONDBLCLK, EVENT_MBUTTONDBLCLK,
#         EVENT_MOUSEMOVE: 

#    flags: EVENT_FLAG_CTRLKEY, EVENT_FLAG_SHIFTKEY, EVENT_FLAG_ALTKEY,
#         EVENT_FLAG_LBUTTON, EVENT_FLAG_RBUTTON,  EVENT_FLAG_MBUTTON

## 自定義滑桿回應函式
def onTrackbar(pos):
    # 讀取 sliders 的資料
    slider1 = cv.getTrackbarPos('weight', 'fusion')
    slider2 = cv.getTrackbarPos('size', 'fusion')
    slider3 = cv.getTrackbarPos('negative', 'fusion')
    # 注意 slider2 不得等於 0
    if slider2 == 0:
        slider2 = 1
    # 計算 im1 縮放後的寬高
    w = int(im1.shape[1] * slider2 / 100)
    h = int(im1.shape[0] * slider2 / 100)
    # 縮放 im1
    im3 = cv.resize(im1, (w, h))
    # 建立一個與 im2 相同大小的黑色影像
    im4 = np.zeros(im2.shape, dtype=np.uint8)
    # 計算 im3 貼上去的位置
    x = int((im2.shape[1] - im3.shape[1]) / 2)
    y = int((im2.shape[0] - im3.shape[0]) / 2)
    # 將 im3 貼到 im4 的正中央
    im4[y:y+im3.shape[0], x:x+im3.shape[1]] = im3
    # 根據 slider1 的數值，用 cv2.addWeighted 對 im2 與 im4 加權混合
    dst = cv.addWeighted(im2, 1 - slider1/100., im4, slider1/100., 0)
    # 負片效果，只處理指定的比例範圍
    dst[:, :int(dst.shape[1] * slider3 / 100.)] = 255 -  dst[:, :int(dst.shape[1] * slider3 / 100.)]
    # 顯示影像
    cv.imshow("fusion", dst)

	
        
## 主程式起始處

########## Level 1: 手繪 ##########
# 建立 400x400 像素的黑色 uint8 格式影像 im1
im1 = np.zeros((400, 400, 3), np.uint8)
# 在"draw"(手繪)視窗內，顯示該影像
cv.namedWindow('draw')
# 用cv2.setMouseCallback建立滑鼠回應函式
cv.setMouseCallback('draw', onMouse)
# 主程式等待按下鍵盤任意鍵
while(1):
    cv.imshow('draw',im1)
    if cv.waitKey():
        break
cv.destroyAllWindows()


########## Level 2 & 3 & 4: 影像融合+大小調整+陰陽調整 (三條滑桿) ##########
# 讀取背景影像im2，影像為 "./ntust.jpg"
im2 = cv.imread('ntust.jpg')
# 在"fusion"視窗，顯示背景影像
cv.namedWindow('fusion')

# 用cv2.createTrackbar建立weight滑桿
cv.createTrackbar('weight', 'fusion', 50, 100, onTrackbar)
# 用cv2.createTrackbar建立size滑桿
cv.createTrackbar('size', 'fusion', 50, 100, onTrackbar)
# 用cv2.createTrackbar建立negative滑桿
cv.createTrackbar('negative', 'fusion', 50, 100, onTrackbar)
# onTrackbar回應函式初始化
onTrackbar(0)
# 加分題：按 Esc 離開，按’r’重置三條滑桿
while(1):
    k = cv.waitKey(1) & 0xFF
    if k == 27: # 如果是 Esc (ASCII 第27號)，脫離迴圈
        break
    elif k==ord('r'): # 重置三條滑桿
         cv.setTrackbarPos('weight','fusion', 50)
         cv.setTrackbarPos('size','fusion', 50)
         cv.setTrackbarPos('negative','fusion', 50)
         onTrackbar(0)        
cv.destroyAllWindows()