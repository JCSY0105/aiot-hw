import cv2

# 加載圖片
img1 = cv2.imread('box.png')
img2 = cv2.imread('box_in_scene.png')

# 確保圖片正確載入
if img1 is None or img2 is None:
    print("無法載入圖片，請檢查檔案路徑")
    exit()

# 初始化特徵檢測器 (改用 SIFT)
feature = cv2.SIFT_create()

# 檢測並計算特徵點與描述子
kp1, des1 = feature.detectAndCompute(img1, None)
kp2, des2 = feature.detectAndCompute(img2, None)

# 使用 BFMatcher 進行特徵匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 篩選優良匹配點
good = []
for m, n in matches:
    if m.distance < 0.55 * n.distance:
        good.append(m)

print('Matching points :{}'.format(len(good)))

# 繪製匹配結果
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 顯示結果
cv2.imshow('video', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()