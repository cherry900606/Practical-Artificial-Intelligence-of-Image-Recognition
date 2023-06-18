import cv2 # 匯入 openCV 套件
import numpy as np # 匯入 Numpy 套件

# 載入哈爾小波級聯正臉偵測訓練集(用CascadeClassifier())
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

# 建立視訊物件並讀取影片檔
cap = cv2.VideoCapture("./sleepy.mp4")

## 讀取視訊參數 ##
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) #視訊畫面高
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) #視訊畫面寬
fps = cap.get(cv2.CAP_PROP_FPS) #視訊幀率
total_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) #視訊總幀數

face_counter = 0 # 計算人臉出現間隔
font = cv2.FONT_HERSHEY_SIMPLEX #字體
option = 1 #人臉特效選項(預設1，不做特效)
frame_index = 0 #畫面編號(預設0)

while True:  # 用無窮迴圈讀取影片中每個畫格(幀)   
    ret, frame = cap.read() # 讀取影片中的畫格
    if ret == True: # 若有讀取到影片中的畫格
        frame_index += 1 # 畫面數量加 1

        ###### 偵測膚色 #######
        # ROI是高寬25%:75%範圍
        ROI = frame[int(height * 0.25) : int(height * 0.75), int(width * 0.25) : int(width * 0.75), :]
        # 將ROI從 BGR 轉換至 HSV 色空間(用 cvtColor())
        hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
        # 定義 HSV (hue, saturation, value) 空間的膚色上下界範圍(下界約 (0,50,50), 上界約 (80,180,220))
        # 註：HSV的範圍上限 [180, 256, 256]
        lower_skin = np.array([0,50,50])
        upper_skin = np.array([80,180,220])
        # 取HSV色空間下，ROI範圍內的膚色遮罩(用inRange())
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        # 算出膚色面積率，也就膚色遮罩非零數值佔遮罩面積的比率(用np.count_nonzero()，可用mask[:]忽略維度，用round(,精度)四捨五入)
        skinAreaRate = round(np.count_nonzero(mask[:]) / (mask.shape[0] * mask.shape[1]), 3)
        # 如果膚色面積率高於0.07，代表「有膚色」，否則代表「無膚色」
        if skinAreaRate > 0.07:
            hasSkin = True
        else:
            hasSkin = False
        # 將膚色遮罩轉換為彩色格式(用cvtColor)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # 在遮罩上加面積率數值(用putText())       
        cv2.putText(mask,"Area: " + str(skinAreaRate),(10,50), font, 1, (255,255,255), 2)
        # 建立跟輸入影像一樣大，一樣格式的背景影像(用np.zeros())
        background = np.zeros((int(height), int(width), 3), dtype='uint8')
        
        # 將背景影像設成灰色
        background[:, :, :] = 128
        # 把膚色遮罩貼入背景影像
        background[int(height * 0.25) : int(height * 0.75), int(width * 0.25) : int(width * 0.75), :] = mask


        ###### 偵測人臉 ######
        # 影像轉成灰階格式(用cvtColor())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #影像轉灰階
        # 偵測正臉(用.detectMultiScale())
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #偵測人臉
        #### 判斷是否打瞌睡 #####
        # 如果「有膚色」但「無正臉」，在視窗中顯示"Wake Up!"字樣(最好能閃爍)
        if hasSkin and len(faces) == 0:
            if face_counter == 10:
                face_counter -= 1
            elif face_counter > 0 and face_counter < 10:
                face_counter -= 1               
                cv2.putText(frame, "Wake Up!",(12,202), font,  1,(0,0,0), 2) #背景黑字
                # 藍黃閃爍判斷(各兩秒)
                if frame_index % 4 < 2:
                    cv2.putText(frame,"Wake Up!",(10,202), font, 1, (255,0,0), 2) #前景字
                else:
                    cv2.putText(frame,"Wake Up!",(10,202), font, 1, (0,255,255), 2) #前景字        
        # 如果「無膚色」且「無正臉」，在視窗中顯示"Nobody"字樣
        elif hasSkin == False and len(faces) ==0:  
            cv2.putText(frame, "Nobody",(12,202), font,  1,(0,0,0), 2) #背景黑字
            cv2.putText(frame,"Nobody",(10,202), font, 1, (0,0,255), 2) #前景白字
       # 否則「有正臉」 
        else :    
            # 識別到臉，就把 counter 重新設回 10
            face_counter = 10
            # 取得人臉的資訊(座標、長寬)
            (x,y,w,h) = faces[0]
            #增加矩形框的高度
            h += 5

            # 如果人臉特效選項為2
            if option == 2: # 二值化
                # 將該區間的圖像轉成灰階
                gray_face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                # 對該區間的圖像進行二值化處理
                ret, binary_face = cv2.threshold(gray_face, 120, 255, cv2.THRESH_BINARY)
                # 將處理後的圖像放回原圖
                frame[y:y+h, x:x+w] = cv2.cvtColor(binary_face, cv2.COLOR_GRAY2BGR)
            # 如果人臉特效選項為3
            elif option == 3: # 負片效果
                frame[y:y+h, x:x+w] = 255 - frame[y:y+h, x:x+w]
            # 如果人臉特效選項為4
            elif option == 4: # 邊緣檢測
                output = cv2.GaussianBlur(gray[y:y+h, x:x+w],(5, 5), 0) # 模糊化，去除雜訊
                output = cv2.Canny(output, 1, 10) # 偵測邊緣
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                frame[y:y+h, x:x+w] = output
            elif option == 5: # 灰階
                output = cv2.cvtColor(gray[y:y+h, x:x+w], cv2.COLOR_GRAY2BGR) # 灰階轉回彩色
                frame[y:y+h, x:x+w] = output

            #繪製人臉矩形框(用rectangle())
            cv2.rectangle(frame,(x,y),(x+h,y+w),(255,0,0),2) #繪製人臉矩形框
            #繪製人臉橢圓形框(用ellipse())
            cv2.ellipse(frame, (int(x + w/2), int(y + w/2)), (int(w/2), int(w/2)), 0, 0, 360, (0, 255, 255), 2)
            #在人臉矩形框上方放學號文字(用putText())
            cv2.putText(frame, "B10832019",(x+2 ,y - 20), font,  1,(0,0,0), 2) #背景黑字
            cv2.putText(frame,"B10832019",(x ,y - 20), font, 1, (0,255,255), 2) #前景字

        #水平合併膚色遮罩與人臉偵測影像(用np.hstack)，並顯示該畫格(用imshow())
        frame = np.hstack((background, frame))
        cv2.imshow("Video", frame) #顯示該畫格
        # 停每800/fps毫秒讀取鍵盤的按鍵
        key = cv2.waitKey(round(800/fps)) & 0xFF
        if key == 27: #當按鍵為ESC(ASCII碼為27)時跳出迴圈
            break   
        # 如果電腦鍵盤按下1(可用ord('1')轉成相應的ASCII碼)，將人臉區域處理選項設為 1(不做特效)
        elif key == ord('1'):
            option = 1
        # 如果電腦鍵盤按下2，將人臉區域處理選項設為2 (執行特效2)
        elif key == ord('2'):
            option = 2
        # 如果電腦鍵盤按下3，將人臉區域處理選項設為3 (執行特效3)
        elif key == ord('3'):
            option = 3
        # 如果電腦鍵盤按下4，將人臉區域處理選項設為4 (執行特效4)
        elif key == ord('4'):
            option = 4
        # 如果電腦鍵盤按下5，將人臉區域處理選項設為4 (執行特效5)
        elif key == ord('5'):
            option = 5
    else: # 當沒讀到影片中的Frame時，跳出迴圈
        break

cap.release()  # 釋放記憶體
cv2.destroyAllWindows()