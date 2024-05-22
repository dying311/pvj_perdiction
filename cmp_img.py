"""
@filename:cmp_img.py
@author:dying
@time:2024-05-22
"""
#图像相似度。实际没用
import cv2


# Hash值对比
def cmpHash(hash1, hash2,shape = (10,10)):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 相等则n计数+1，n最终为相似度
        if hash1[i] == hash2[i]:
            n = n + 1
    return n / (shape[0] * shape[1])


# 均值哈希算法
def aHash(img, shape = (10,10)):
    # 缩放为10*10
    img = cv2.resize(img, shape)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(shape[0]):
        for j in range(shape[1]):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 100
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


img1 = cv2.imread("C:\\Users\\xiong\\Pictures\\image_1.png")
img2 = cv2.imread("D:\\Desktop\\strom_prediction\\strom\pvj\\test_pvj_231.jpg")
shape = (366, 366)
hash1 = aHash(img1)
hash2 = aHash(img2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)
