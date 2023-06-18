import numpy as np
import cv2 as cv
from tensorflow import keras
from keras import layers

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
			cv.circle(drawing, (x,y), 6, (255,255,255), -1)
            #顯示影像
			cv.imshow("draw", drawing)
	#如果滑鼠左鍵彈起
	elif event == cv.EVENT_LBUTTONUP:
		#buttonDown設為 False
		buttonDown = False
	#如果按下滑鼠右鍵
	elif event == cv.EVENT_RBUTTONDOWN:
		#將視窗刪除
		cv.destroyWindow("draw")
		

isEnd = False
model = keras.models.load_model('model_2.h5')

while isEnd != True:
	drawing = np.zeros((150, 150, 3), np.uint8)
	cv.namedWindow('draw')
	cv.setMouseCallback('draw', onMouse)
	while(1):
		cv.imshow('draw',drawing)
		if cv.waitKey() == ord('q'):
			isEnd = True
		if cv.waitKey():
			break
	cv.destroyAllWindows()
	if isEnd:
		break
	drawing = cv.cvtColor(drawing, cv.COLOR_BGR2GRAY)
	im = cv.resize(drawing, (32, 32))
	label = model.predict(im.reshape(1, 32, 32))
	pred = np.argmax(label)
	if pred >= 10:
		pred = chr(pred - 10 + ord('A')) 
	else:
		pred = chr(pred + ord('0'))
	while(1):
		cv.putText(drawing, pred, (25, 25), 
	     cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
		cv.imshow('show',drawing)
		if cv.waitKey():
			break
	cv.destroyAllWindows()