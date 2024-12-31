import cv2
import numpy as np

# 加載圖像
image = cv2.imread('blox.jpg')
if image is None:
    raise FileNotFoundError("無法加載圖像，請確保 'blox.jpg' 文件存在於當前目錄中。")

# 初始化特徵檢測算法
try:
    sift_feature = cv2.SIFT_create()
    surf_feature = cv2.xfeatures2d.SURF_create()  # 需要 opencv-contrib-python
except AttributeError:
    raise ImportError("請確保已安裝 'opencv-contrib-python' 並檢查 OpenCV 版本是否支持 SIFT/SURF。")

orb_feature = cv2.ORB_create()

# 檢測特徵點和計算描述符
sift_kp, _ = sift_feature.detectAndCompute(image, None)
surf_kp, _ = surf_feature.detectAndCompute(image, None)
orb_kp, _ = orb_feature.detectAndCompute(image, None)

# 繪製特徵點
sift_out = cv2.drawKeypoints(image, sift_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
surf_out = cv2.drawKeypoints(image, surf_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
orb_out = cv2.drawKeypoints(image, orb_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 合併圖像
output = cv2.vconcat([
    cv2.hconcat([image, sift_out]),
    cv2.hconcat([surf_out, orb_out])
])

# 顯示結果
cv2.imshow('Feature Detection Comparison', output)
cv2.waitKey(0)
cv2.destroyAllWindows()