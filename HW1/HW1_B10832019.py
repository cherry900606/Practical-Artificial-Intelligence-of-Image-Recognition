#作業1: python, np, plt 繪圖應用
#載入相關套件模組
import numpy as np
import matplotlib.pyplot as plt


#輸入影像檔名字串 filename (用 input)
filename = input('輸入影像檔名字串:')

## Part 1: 繪製 Colorbars ################
#生成0~255的一維整數陣列a (用 numpy)
a = np.arange(256)
#格式改為uint8 (用 astype)
a = a.astype(dtype = np.uint8)
#a重複3次，長度成為256*3的b陣列 (用 repeat)
b = np.repeat(a, 3)
#建立(64*8,256*3,3)大小的零矩陣c，格式是uint8
c = np.zeros((64*8,256*3,3), dtype=np.uint8)

    
#用兩層for迴圈產生影像資訊：第一層跟高有關，第二層跟RGB通道有關
# 第一層重複 8 次，代表該圖中的 8 個 bar
# 第二層重複 3 次，代表 R, G, B 的 3 色通道
for i in range(8):
    for j in range(3):
        #將b複製到c矩陣特定掃描線與色通道
        #可用np.invert將陣列b的順序倒置

        # 藉由 i 的奇偶數選擇 b or invert(b)
        if i % 2 == 0:
            row = b;
        else:
            row = np.invert(b)
        # 藉由 i 值，也就是第幾個 bar，決定是否要給予該 RGB 通道
        # 由於以 bar 為高的單位，所以要乘上 64，因為每個 bar 高為 64
        if j == 0 and (i in [0, 1, 2, 3]):
            c[i*64:(i+1)*64, :, j] = row
        elif j == 1 and (i in [0, 1, 4, 5]):
            c[i*64:(i+1)*64, :, j] = row
        elif j == 2 and (i in [0, 1, 6, 7]):
            c[i*64:(i+1)*64, :, j] = row


#建立1號視窗(用 figure)
figure1 = plt.figure(1)
#顯示影像c(用 imshow)
plt.imshow(c)
#加標題
plt.title("Fig.1 Colorbars")
#不顯示 x,y 軸刻度值
plt.axis('off')
#暫停2秒(用 pause)
plt.pause(2)
#關閉1號視窗(用 close)
plt.close()

# Part 2: 繪製黑白圈圈圖 #################
# 自訂一個函數，名稱是 radius，輸入u,v影像座標，算出(return)遠離中心的半徑
def radius(u, v):
    return np.sqrt(u ** 2 + v** 2)

#產生一個6至1之間，以-0.2為間隔的數列d
d = np.arange(6, 1, -0.2)
#將d平方，再用cumsum產生半徑的門檻值數列e
e = np.cumsum(d ** 2)

#以float32格式，建立512x512大小的零矩陣f
f = np.zeros((512, 512), dtype=np.float32)

# 該方法背後的邏輯是將黑色與白色圓圈看作介於某範圍間的圖形
# 因此透過事先定義好的門檻值(e)，決定某點屬於第幾個圓圈，進而決定是黑色還是白色
#利用雙層for迴圈，窮舉x,y影像座標
for x in range(512):
    for y in range(512):
        #用radius函式計算遠離影像中心的半徑r，x,y要先減去255
        r = radius(x - 255, y - 255)
        idx = 0
        #用for迴圈，依序查詢數列e裡的數值
        for z in range(len(e)):
        #如果數列e裡的第z筆數值大於r，就用變數idx記錄該索引值z，並用(break)離開for迴圈
            if e[z] > r:
                idx = z
                break
        #將idx值取除以2的餘數，將該數值存入影像f的x,y位置
        f[x, y] = (idx % 2) * 255

#建立2號視窗(用 figure)
figure2 = plt.figure(2)
#顯示影像f，色彩對用表用'gray'(用 imshow, 以及cmap參數)
plt.imshow(f, cmap='gray')
#加標題
plt.title("Figure.2 Rings")
#水平軸刻度用0~512, 以64為間隔(用 xtick)
plt.xticks(np.arange(0,513,64))
#垂直軸刻度用0~512, 以64為間隔(用 ytick)
plt.yticks(np.arange(0,513,64))
#暫停2秒(用 pause)
plt.pause(2)
#關閉2號視窗(用 close)
plt.close(fig=figure2)

# Part 3: 6種影像處理之水平拼接圖像，隨機排序+輪播 #########
# 自訂一個函數，名稱是 process
# 輸入正方形彩色影像(im_in)以及影像處理選項p
# 根據選項 p, 將處理後的影像(im_out)輸出(return)
# 可以考慮的處理有 rot90, fliplr, flipud, bitwise_not, clip, where 等等
def process(im_in, p):
    if p==0: # 向左選轉 90 度
       im_out = np.rot90(im_in, 1)
    elif p==1: # 將所有數值大於 150 的像素都設為 150
       im_out = np.where(im_in > 150, im_in, 150)
    elif p == 2: # 左右翻轉
        im_out = np.fliplr(im_in)
    elif p == 3: # 上下翻轉
       im_out = np.flipud(im_in)
    elif p == 4: # 把每個像素的二進位 bit 零變一、一變零
       im_out = np.bitwise_not(im_in)
    elif p == 5: # 將像素範圍限縮在 0 到 150
       im_out = np.clip(im_in, 0, 150)
    return im_out

#主程式進入點
#讀取 filename 指定的彩色影像im1 (用 plt.imread)
im1 = plt.imread(filename)
#讀取該影像的高(h)/寬(w)/通道數(ch) (用.shape)
h = im1.shape[0]
w = im1.shape[1]
ch = im1.shape[2]

#取影像im1的局部，使im2的高寬都是h
im2 = im1[:h, :h, :]
#建立uint8格式的四維零陣列，尺寸是(6,h,h,ch)
im3 = np.zeros((6, h, h, ch), dtype=np.uint8)
#建立一個 for 迴圈，分別把 process 產生的6種影像，存到im3裡
for i in range(6):
    #im3的第一維就是process的選項
    im3[i] = process(im2, i)

#用 for 迴圈跑五次隨機順序排列的影像im4
for i in range(5):
    #用np.random.permutation(6)產生0~5的隨機序列
    g = np.random.permutation(6)
    #用 hstack, 以g的順序，水平合併6個子圖
    im4 = np.hstack((im3[g[0]], im3[g[1]], im3[g[2]], im3[g[3]], im3[g[4]], im3[g[5]]))
    
    #建立3號視窗(用 figure)
    figure3 = plt.figure(3)
    #顯示影像im4(用 imshow)
    plt.imshow(im4)
    #加標題(含隨機次數)
    plt.title("Random order: " + str(i))
    #不顯示 x,y 軸刻度值
    plt.axis('off')
    #暫停1秒
    plt.pause(1)
#將im4存入'HW1.jpg'檔
plt.imsave('HW1.jpg', im4)
#關閉3號視窗(用 close)
plt.close()

## Part 4: 將最後的6個子圖，以2x3圖形陣列(用add_subplot)，全螢幕顯示 ####
#建立4號視窗(用 figure)
figure4 = plt.figure(4)
#建立 for 迴圈，依序處理圖形陣列中的6個圖
for i in range(6):
    #添加2x3圖形陣列的第i+1個子圖
    figure4.add_subplot(2, 3, i+1)
    #讀取im3中g[i]序號所對映的子影像，並用squeeze降成三維，存入im5
    im5 = np.squeeze(im3[g[i]])
    #顯示影像im5
    plt.imshow(im5)
    #顯示子圖的標題
    plt.title('Proc.' + str(g[i] + 1))
    #不顯示 x,y 軸刻度值
    plt.axis('off')

#全螢幕顯示
figManager = plt.get_current_fig_manager()
figManager.window.state('zoomed')
plt.show() #顯示




