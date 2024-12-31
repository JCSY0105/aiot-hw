# 匯入 OpenCV 模組
import cv2

# 加載 Haar 紋波檢測器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# 讀取圖片並轉換為灰階
image = cv2.imread('demo.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 檢測臉部
# 第一個參數是灰階圖像，第二個參數是縮放因子（每次檢測時圖像尺寸縮小的比例），第三個參數是最小鄰近數
faces = face_cascade.detectMultiScale(gray, 1.05, 20)

# 遍歷所有檢測到的臉部區域，並繪製綠色矩形框
for (x, y, w, h) in faces:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# 創建可調整大小的視窗，並顯示結果
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.imshow('video', image)

# 等待用戶按下任意鍵關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()