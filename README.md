## 一、高空抛物检测

### 1.项目应用场景

![https://gitee.com/livingbody/gkpw/raw/master/%E9%AB%98%E7%A9%BA%E6%8A%9B%E7%89%A91](https://gitee.com/livingbody/gkpw/raw/master/%E9%AB%98%E7%A9%BA%E6%8A%9B%E7%89%A91)

![https://gitee.com/livingbody/gkpw/raw/master/%E9%AB%98%E7%A9%BA%E6%8A%9B%E7%89%A9](https://gitee.com/livingbody/gkpw/raw/master/%E9%AB%98%E7%A9%BA%E6%8A%9B%E7%89%A9)

“出门戴上头盔吧，怕砸”

“天上不会掉馅饼，但会掉锅碗瓢盆”

“一经过高层楼下就快步通过”

……

### 2.人工智能抛物检测

这“高空抛物”可不让人生省心

啥东西都敢掉下来

更让人气愤的是

肇事者抛物的理由也是千奇百怪

心情不好，抛

和男女朋友吵架，抛

觉得有事看不顺眼，抛

在他们看来

总之一句话，“想抛就抛”

你以为抓不到你吗？

错了！

现在要cue高空抛物者了

“你们的一举一动，已经被记录下来了”

### 3.基本思路

- 计算compare_ssim

- 计算出异常的位置进行抠图

- 利用pp-shitu对发现的物体进行分类识别

- 对3秒钟存在告警且识别的物体进行报警。



## 二、异常物体检测

### 1.基本情况

![](https://ai-studio-static-online.cdn.bcebos.com/fcbc5dd4ddd24ca19e3d7fdf3df1de035e731662370d46dc8a302d92177f8a0f)

第一帧

![](https://ai-studio-static-online.cdn.bcebos.com/7d20011f898a4374b26f4c7c95054917eb6773d2764c491a9693587f3052262b)

第二帧

![](https://ai-studio-static-online.cdn.bcebos.com/1efd1d18407d4beab81273191e04e306c7573aa6489b4802b79d5bd5aadc6498)

检测出的物体


### 2.计算步骤
```
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2


#加载两张图片：
#注意，从文件路径复制来的斜杠是反的，记得更改，且用英文路径

imageA = cv2.imread("gl_1.jpeg")
imageB = cv2.imread("gl_2.jpeg")

#将他们转换为灰度：

grayA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)


#计算两个灰度图像之间的结构相似度指数：
#不过ssim多用于压缩图片后的失真度比较。。

(score,diff) = compare_ssim(grayA,grayB,full = True)
diff = (diff *255).astype("uint8")



#找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形：

thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。
#其首先返回一个list，list中每个元素都是图像中的一个轮廓

cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


"""注意cv版本，下面这一行会出现下列问题：
OpenCV 3 改为cv2.findContours(...)返回值为image, contours, hierarchy，

OpenCV 2 cv2.findContours(...)和OpenCV 4 的cv2.findContours(...)返回值为contours, hierarchy。"""

#把contour轮廓储存在cnts这个list列表里

cnts = cnts[1] if imutils.is_cv2() else cnts[0]


#找到一系列区域，在区域周围放置矩形：
"""

cv2.rectangle(imageA,(x,y),(x+w,y+h),(0,0,255),2)  参数解释

第一个参数：img是原图

第二个参数：（x，y）是矩阵的左上点坐标

第三个参数：（x+w，y+h）是矩阵的右下点坐标

第四个参数：（0,0,255）是画线对应的rgb颜色

第五个参数：2是所画的线的宽度
"""

for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(imageA,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.rectangle(imageB,(x,y),(x+w,y+h),(0,0,255),2)
    ex_obj=imageB[y:y+h,x:x+w]
    cv2.imwrite('ex_obj.jpg',ex_obj)




#用cv2.imshow 展现最终对比之后的图片， cv2.imwrite 保存最终的结果图片

cv2.imshow("differ",imageB)
cv2.imwrite("differ.jpg",imageB)
cv2.waitKey(0)

```


```python
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2


#加载两张图片：
#注意，从文件路径复制来的斜杠是反的，记得更改，且用英文路径

imageA = cv2.imread("gl_1.jpeg")
imageB = cv2.imread("gl_2.jpeg")

#将他们转换为灰度：

grayA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)


#计算两个灰度图像之间的结构相似度指数：
#不过ssim多用于压缩图片后的失真度比较。。

(score,diff) = compare_ssim(grayA,grayB,full = True)
diff = (diff *255).astype("uint8")



#找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形：

thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。
#其首先返回一个list，list中每个元素都是图像中的一个轮廓

cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


"""注意cv版本，下面这一行会出现下列问题：
OpenCV 3 改为cv2.findContours(...)返回值为image, contours, hierarchy，

OpenCV 2 cv2.findContours(...)和OpenCV 4 的cv2.findContours(...)返回值为contours, hierarchy。"""

#把contour轮廓储存在cnts这个list列表里

cnts = cnts[1] if imutils.is_cv2() else cnts[0]


#找到一系列区域，在区域周围放置矩形：
"""

cv2.rectangle(imageA,(x,y),(x+w,y+h),(0,0,255),2)  参数解释

第一个参数：img是原图

第二个参数：（x，y）是矩阵的左上点坐标

第三个参数：（x+w，y+h）是矩阵的右下点坐标

第四个参数：（0,0,255）是画线对应的rgb颜色

第五个参数：2是所画的线的宽度
"""

for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(imageA,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.rectangle(imageB,(x,y),(x+w,y+h),(0,0,255),2)
    ex_obj=imageB[y:y+h,x:x+w]
    cv2.imwrite('ex_obj.jpg',ex_obj)




#用cv2.imshow 展现最终对比之后的图片， cv2.imwrite 保存最终的结果图片

cv2.imshow("differ",imageB)
cv2.imwrite("differ.jpg",imageB)
cv2.waitKey(0)

```

## 三、pp-shitu进行物体分类

### 1.环境配置
下载PaddleClas：下载官方repo的PaddleClas代码


```python
!git clone https://gitee.com/PaddlePaddle/PaddleClas --depth=1
```

    Cloning into 'PaddleClas'...
    remote: Enumerating objects: 1413, done.[K
    remote: Counting objects: 100% (1413/1413), done.[K
    remote: Compressing objects: 100% (1009/1009), done.[K
    remote: Total 1413 (delta 566), reused 837 (delta 378), pack-reused 0[K
    Receiving objects: 100% (1413/1413), 61.53 MiB | 4.68 MiB/s, done.
    Resolving deltas: 100% (566/566), done.
    Checking connectivity... done.



```python
# 大约耗时40秒
!pip install pip -U 
!cd PaddleClas && pip install -r requirements.txt
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting pip
    [?25l  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a4/6d/6463d49a933f547439d6b5b98b46af8742cc03ae83543e4d7688c2420f8b/pip-21.3.1-py3-none-any.whl (1.7MB)
    [K    100% |████████████████████████████████| 1.7MB 15.5MB/s ta 0:00:01
    [?25hInstalling collected packages: pip
      Found existing installation: pip 19.0.3
        Uninstalling pip-19.0.3:
          Successfully uninstalled pip-19.0.3
    Successfully installed pip-21.3.1
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Requirement already satisfied: prettytable in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 1)) (0.7.2)
    Requirement already satisfied: ujson in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 2)) (1.35)
    Collecting opencv-python==4.4.0.46
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6d/80/10a9ae6fa0940f25af32739d1dc6dfdbbdc79af3f04c5ea1a6de4303cd54/opencv_python-4.4.0.46-cp36-cp36m-manylinux2014_x86_64.whl (49.5 MB)
         |████████████████████████████████| 49.5 MB 329 kB/s            
    [?25hRequirement already satisfied: pillow in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 4)) (5.4.1)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 5)) (4.32.2)
    Requirement already satisfied: PyYAML in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 6)) (5.1)
    Collecting visualdl>=2.2.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/87/c8/10d0d24822637d8e5493a73ad118640530195e45b1c71ae0e60606ff5f0e/visualdl-2.2.3-py3-none-any.whl (2.7 MB)
         |████████████████████████████████| 2.7 MB 50.5 MB/s            
    [?25hRequirement already satisfied: scipy in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 8)) (1.5.3)
    Requirement already satisfied: scikit-learn==0.23.2 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 9)) (0.23.2)
    Collecting gast==0.3.3
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d6/84/759f5dd23fec8ba71952d97bcc7e2c9d7d63bdc582421f3cd4be845f0c98/gast-0.3.3-py2.py3-none-any.whl (9.7 kB)
    Collecting faiss-cpu==1.7.1.post2
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5f/2e/a7c8358aeedd28c6026890b4f292498e0440118cf4d32398d2647e1216c6/faiss_cpu-1.7.1.post2-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.4 MB)
         |████████████████████████████████| 8.4 MB 54.8 MB/s            
    [?25hRequirement already satisfied: numpy>=1.13.3 in /opt/conda/lib/python3.6/site-packages (from opencv-python==4.4.0.46->-r requirements.txt (line 3)) (1.19.4)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/lib/python3.6/site-packages (from scikit-learn==0.23.2->-r requirements.txt (line 9)) (1.0.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.6/site-packages (from scikit-learn==0.23.2->-r requirements.txt (line 9)) (2.1.0)
    Collecting pillow
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ea/0f/2fa195c2d8c6fe0b3dc2df5fc6ac6b8dbd005ea30aaa0fa43eca88b8c664/Pillow-8.4.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
         |████████████████████████████████| 3.1 MB 46.9 MB/s            
    [?25hRequirement already satisfied: pandas in /opt/conda/lib/python3.6/site-packages (from visualdl>=2.2.0->-r requirements.txt (line 7)) (0.24.2)
    Requirement already satisfied: matplotlib in /opt/conda/lib/python3.6/site-packages (from visualdl>=2.2.0->-r requirements.txt (line 7)) (2.2.3)
    Collecting bce-python-sdk
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/55/ef/2a7e6c7692a036bae2570a9bcdcd7963ea54e07db97b4554c24d3cfacb21/bce-python-sdk-0.8.64.tar.gz (127 kB)
         |████████████████████████████████| 127 kB 72.6 MB/s            
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting shellcheck-py
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0f/4f/ab756db996bdde0a647bf552c1be78cbf8055b664ecd54f08f3210f8cf26/shellcheck_py-0.8.0.3-py2.py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
         |████████████████████████████████| 2.1 MB 46.6 MB/s            
    [?25hRequirement already satisfied: requests in /opt/conda/lib/python3.6/site-packages (from visualdl>=2.2.0->-r requirements.txt (line 7)) (2.22.0)
    Collecting protobuf>=3.11.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0f/1c/6b3b5b8c07e92b84cb7d4fb946a8bc72b98d93d8b7c8e8a8e45023745810/protobuf-3.19.4-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
         |████████████████████████████████| 1.1 MB 49.6 MB/s            
    [?25hCollecting flask>=1.1.1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cd/77/59df23681f4fd19b7cbbb5e92484d46ad587554f5d490f33ef907e456132/Flask-2.0.3-py3-none-any.whl (95 kB)
         |████████████████████████████████| 95 kB 14.3 MB/s            
    [?25hRequirement already satisfied: flake8>=3.7.9 in /opt/conda/lib/python3.6/site-packages (from visualdl>=2.2.0->-r requirements.txt (line 7)) (3.8.2)
    Collecting Flask-Babel>=1.0.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ab/3e/02331179ffab8b79e0383606a028b6a60fb1b4419b84935edd43223406a0/Flask_Babel-2.0.0-py3-none-any.whl (9.3 kB)
    Collecting pre-commit
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d6/a0/9c06353771c8dae6db437dd513a885eccdb1566cb332569130484eddf4e7/pre_commit-2.17.0-py2.py3-none-any.whl (195 kB)
         |████████████████████████████████| 195 kB 57.8 MB/s            
    [?25hCollecting six>=1.14.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d9/5a/e7c31adbe875f2abbb91bd84cf2dc52d792b5a01506781dbcf25c91daf11/six-1.16.0-py2.py3-none-any.whl (11 kB)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/lib/python3.6/site-packages (from flake8>=3.7.9->visualdl>=2.2.0->-r requirements.txt (line 7)) (2.2.0)
    Requirement already satisfied: importlib-metadata in /opt/conda/lib/python3.6/site-packages (from flake8>=3.7.9->visualdl>=2.2.0->-r requirements.txt (line 7)) (1.6.1)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/lib/python3.6/site-packages (from flake8>=3.7.9->visualdl>=2.2.0->-r requirements.txt (line 7)) (2.6.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/lib/python3.6/site-packages (from flake8>=3.7.9->visualdl>=2.2.0->-r requirements.txt (line 7)) (0.6.1)
    Collecting Jinja2>=3.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/20/9a/e5d9ec41927401e41aea8af6d16e78b5e612bca4699d417f646a9610a076/Jinja2-3.0.3-py3-none-any.whl (133 kB)
         |████████████████████████████████| 133 kB 62.3 MB/s            
    [?25hCollecting Werkzeug>=2.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f4/f3/22afbdb20cc4654b10c98043414a14057cd27fdba9d4ae61cea596000ba2/Werkzeug-2.0.3-py3-none-any.whl (289 kB)
         |████████████████████████████████| 289 kB 45.3 MB/s            
    [?25hCollecting click>=7.1.2
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4a/a8/0b2ced25639fb20cc1c9784de90a8c25f9504a7f18cd8b5397bd61696d7d/click-8.0.4-py3-none-any.whl (97 kB)
         |████████████████████████████████| 97 kB 1.8 MB/s             
    [?25hCollecting itsdangerous>=2.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9c/96/26f935afba9cd6140216da5add223a0c465b99d0f112b68a4ca426441019/itsdangerous-2.0.1-py3-none-any.whl (18 kB)
    Collecting Babel>=2.3
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/aa/96/4ba93c5f40459dc850d25f9ba93f869a623e77aaecc7a9344e19c01942cf/Babel-2.9.1-py2.py3-none-any.whl (8.8 MB)
         |████████████████████████████████| 8.8 MB 2.2 MB/s            
    [?25hRequirement already satisfied: pytz in /opt/conda/lib/python3.6/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.2.0->-r requirements.txt (line 7)) (2018.9)
    Collecting pycryptodome>=3.8.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/71/6b/965be3bec8fc1f6013739fea2bba86cf43a3be09ab7e29bd2134829c7615/pycryptodome-3.14.1-cp35-abi3-manylinux2010_x86_64.whl (2.0 MB)
         |████████████████████████████████| 2.0 MB 63.9 MB/s            
    [?25hRequirement already satisfied: future>=0.6.0 in /opt/conda/lib/python3.6/site-packages (from bce-python-sdk->visualdl>=2.2.0->-r requirements.txt (line 7)) (0.17.1)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.6/site-packages (from matplotlib->visualdl>=2.2.0->-r requirements.txt (line 7)) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/lib/python3.6/site-packages (from matplotlib->visualdl>=2.2.0->-r requirements.txt (line 7)) (2.3.1)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/lib/python3.6/site-packages (from matplotlib->visualdl>=2.2.0->-r requirements.txt (line 7)) (2.8.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.6/site-packages (from matplotlib->visualdl>=2.2.0->-r requirements.txt (line 7)) (1.0.1)
    Collecting nodeenv>=0.11.1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/54/73/56c89b343befb9c63e8117294d265458f0ff726fa2abcdc6bb5ec5e66a1a/nodeenv-1.6.0-py2.py3-none-any.whl (21 kB)
    Collecting importlib-resources<5.3
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/11/8e/84a6a778a1160cefcef1192a7bd26e4e6689981553aff13c2b2b6f1c352f/importlib_resources-5.2.3-py3-none-any.whl (27 kB)
    Requirement already satisfied: toml in /opt/conda/lib/python3.6/site-packages (from pre-commit->visualdl>=2.2.0->-r requirements.txt (line 7)) (0.10.1)
    Collecting virtualenv>=20.0.8
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/56/a2/3e5fdac9ecca6a3a6d2f63f7a486afd4a72728ba9f2ae83fa43f7af8ac8b/virtualenv-20.13.2-py2.py3-none-any.whl (8.7 MB)
         |████████████████████████████████| 8.7 MB 60.4 MB/s            
    [?25hCollecting cfgv>=2.0.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6d/82/0a0ebd35bae9981dea55c06f8e6aaf44a49171ad798795c72c6f64cba4c2/cfgv-3.3.1-py2.py3-none-any.whl (7.3 kB)
    Collecting identify>=1.0.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/da/1a/93ac674fee1a5af11bdbc1cd895895a8710aa49402558bf91ec3523f0214/identify-2.4.4-py2.py3-none-any.whl (98 kB)
         |████████████████████████████████| 98 kB 19.3 MB/s            
    [?25hRequirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests->visualdl>=2.2.0->-r requirements.txt (line 7)) (2020.12.5)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests->visualdl>=2.2.0->-r requirements.txt (line 7)) (2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests->visualdl>=2.2.0->-r requirements.txt (line 7)) (1.24.1)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests->visualdl>=2.2.0->-r requirements.txt (line 7)) (3.0.4)
    Requirement already satisfied: zipp>=3.1.0 in /opt/conda/lib/python3.6/site-packages (from importlib-resources<5.3->pre-commit->visualdl>=2.2.0->-r requirements.txt (line 7)) (3.1.0)
    Collecting MarkupSafe>=2.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e2/a9/eafee9babd4b3aed918d286fbe1c20d1a22d347b30d2bddb3c49919548fa/MarkupSafe-2.0.1-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (30 kB)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.6/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl>=2.2.0->-r requirements.txt (line 7)) (40.8.0)
    Collecting platformdirs<3,>=2
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b1/78/dcfd84d3aabd46a9c77260fb47ea5d244806e4daef83aa6fe5d83adb182c/platformdirs-2.4.0-py3-none-any.whl (14 kB)
    Collecting filelock<4,>=3.2
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/84/ce/8916d10ef537f3f3b046843255f9799504aa41862bfa87844b9bdc5361cd/filelock-3.4.1-py3-none-any.whl (9.9 kB)
    Collecting distlib<1,>=0.3.1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ac/a3/8ee4f54d5f12e16eeeda6b7df3dfdbda24e6cc572c86ff959a4ce110391b/distlib-0.3.4-py2.py3-none-any.whl (461 kB)
         |████████████████████████████████| 461 kB 61.7 MB/s            
    [?25hCollecting dataclasses
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fe/ca/75fac5856ab5cfa51bbbcefa250182e50441074fdc3f803f6e76451fab43/dataclasses-0.8-py3-none-any.whl (19 kB)
    Building wheels for collected packages: bce-python-sdk
      Building wheel for bce-python-sdk (setup.py) ... [?25ldone
    [?25h  Created wheel for bce-python-sdk: filename=bce_python_sdk-0.8.64-py3-none-any.whl size=202974 sha256=5be01a81f4e003a2983a3e3726049edd2a03aa433b07a39a1a7d5f695c7161f8
      Stored in directory: /home/aistudio/.cache/pip/wheels/4d/01/4a/0ddb9526bd30d94e4ba9d333ed1f462272e55724ee22ea338d
    Successfully built bce-python-sdk
    Installing collected packages: MarkupSafe, dataclasses, Werkzeug, six, platformdirs, Jinja2, itsdangerous, importlib-resources, filelock, distlib, click, virtualenv, pycryptodome, nodeenv, identify, flask, cfgv, Babel, shellcheck-py, protobuf, pre-commit, pillow, Flask-Babel, bce-python-sdk, visualdl, opencv-python, gast, faiss-cpu
      Attempting uninstall: MarkupSafe
        Found existing installation: MarkupSafe 1.1.1
        Uninstalling MarkupSafe-1.1.1:
          Successfully uninstalled MarkupSafe-1.1.1
      Attempting uninstall: Werkzeug
        Found existing installation: Werkzeug 0.15.1
        Uninstalling Werkzeug-0.15.1:
          Successfully uninstalled Werkzeug-0.15.1
      Attempting uninstall: six
        Found existing installation: six 1.12.0
        Uninstalling six-1.12.0:
          Successfully uninstalled six-1.12.0
      Attempting uninstall: Jinja2
        Found existing installation: Jinja2 2.10
        Uninstalling Jinja2-2.10:
          Successfully uninstalled Jinja2-2.10
      Attempting uninstall: itsdangerous
        Found existing installation: itsdangerous 1.1.0
        Uninstalling itsdangerous-1.1.0:
          Successfully uninstalled itsdangerous-1.1.0
      Attempting uninstall: click
        Found existing installation: Click 7.0
        Uninstalling Click-7.0:
          Successfully uninstalled Click-7.0
      Attempting uninstall: flask
        Found existing installation: Flask 1.0.2
        Uninstalling Flask-1.0.2:
          Successfully uninstalled Flask-1.0.2
      Attempting uninstall: protobuf
        Found existing installation: protobuf 3.6.1
        Uninstalling protobuf-3.6.1:
          Successfully uninstalled protobuf-3.6.1
      Attempting uninstall: pillow
        Found existing installation: Pillow 5.4.1
        Uninstalling Pillow-5.4.1:
          Successfully uninstalled Pillow-5.4.1
      Attempting uninstall: opencv-python
        Found existing installation: opencv-python 4.0.0.21
        Uninstalling opencv-python-4.0.0.21:
          Successfully uninstalled opencv-python-4.0.0.21
    Successfully installed Babel-2.9.1 Flask-Babel-2.0.0 Jinja2-3.0.3 MarkupSafe-2.0.1 Werkzeug-2.0.3 bce-python-sdk-0.8.64 cfgv-3.3.1 click-8.0.4 dataclasses-0.8 distlib-0.3.4 faiss-cpu-1.7.1.post2 filelock-3.4.1 flask-2.0.3 gast-0.3.3 identify-2.4.4 importlib-resources-5.2.3 itsdangerous-2.0.1 nodeenv-1.6.0 opencv-python-4.4.0.46 pillow-8.4.0 platformdirs-2.4.0 pre-commit-2.17.0 protobuf-3.19.4 pycryptodome-3.14.1 shellcheck-py-0.8.0.3 six-1.16.0 virtualenv-20.13.2 visualdl-2.2.3


### 2. 图像识别体验
轻量级通用主体检测模型与轻量级通用识别模型和配置文件下载方式如下表所示。
| 模型      | 模型结构   | 预训练模型下载地址   | inference 模型下载地址  | mAP | inference 模型大小(MB) | 单张图片预测耗时(不包含预处理)(ms) |
| :------------:  | :-------------: | :------: | :-------: | :--------: | :-------: | :--------: |
| 轻量级主体检测模型 | PicoDet | [地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_pretrained.pdparams) | [tar 格式文件地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar) [zip 格式文件地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.zip) | 40.1% | 30.1 | 29.8  |
| 服务端主体检测模型 | PP-YOLOv2 | [地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/ppyolov2_r50vd_dcn_mainbody_v1.0_pretrained.pdparams) | [tar 格式文件地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar) [zip 格式文件地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.zip) | 42.5% | 210.5 | 466.6  |

- 可以按照下面的命令下载并解压数据与模型
```
mkdir models
cd models
# 下载识别inference模型并解压
wget {模型下载链接地址} && tar -xf {压缩包的名称}
cd ..

# 下载demo数据并解压
wget {数据下载链接地址} && tar -xf {压缩包的名称}
```

- 使用下面的命令将默认工作目录切换到PaddleClas的deploy文件夹下

```
# import os
# os.chdir("/home/aistudio/PaddleClas/deploy")
# !pwd
%cd ~/PaddleClas/deploy
/home/aistudio/PaddleClas/deploy
```

- 下载、解压 inference 模型与 demo 数据
下载demo数据集以及通用检测、识别模型，命令如下。



```python
%cd ~/PaddleClas/
# !mkdir models
# !cd models && wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar && tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
!wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/general_PPLCNet_x2_5_pretrained_v1.0.pdparams
```

    /home/aistudio/PaddleClas
    --2022-02-27 21:27:31--  https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/general_PPLCNet_x2_5_pretrained_v1.0.pdparams
    Resolving paddle-imagenet-models-name.bj.bcebos.com (paddle-imagenet-models-name.bj.bcebos.com)... 182.61.200.229, 182.61.200.195, 2409:8c04:1001:1002:0:ff:b001:368a
    Connecting to paddle-imagenet-models-name.bj.bcebos.com (paddle-imagenet-models-name.bj.bcebos.com)|182.61.200.229|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 792851195 (756M) [application/octet-stream]
    Saving to: ‘general_PPLCNet_x2_5_pretrained_v1.0.pdparams’
    
    general_PPLCNet_x2_ 100%[===================>] 756.12M  49.4MB/s    in 19s     
    
    2022-02-27 21:27:50 (40.6 MB/s) - ‘general_PPLCNet_x2_5_pretrained_v1.0.pdparams’ saved [792851195/792851195]
    



```python
!cd models && wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar && tar -xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
```

    --2022-02-27 21:27:50--  https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
    Resolving paddle-imagenet-models-name.bj.bcebos.com (paddle-imagenet-models-name.bj.bcebos.com)... 182.61.200.195, 182.61.200.229, 2409:8c04:1001:1002:0:ff:b001:368a
    Connecting to paddle-imagenet-models-name.bj.bcebos.com (paddle-imagenet-models-name.bj.bcebos.com)|182.61.200.195|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 34242560 (33M) [application/x-tar]
    Saving to: ‘general_PPLCNet_x2_5_lite_v1.0_infer.tar’
    
    general_PPLCNet_x2_ 100%[===================>]  32.66M  17.2MB/s    in 1.9s    
    
    2022-02-27 21:27:53 (17.2 MB/s) - ‘general_PPLCNet_x2_5_lite_v1.0_infer.tar’ saved [34242560/34242560]
    



```python
!tree models/ 
```

    models/
    ├── general_PPLCNet_x2_5_lite_v1.0_infer
    │   ├── inference.pdiparams
    │   ├── inference.pdiparams.info
    │   └── inference.pdmodel
    ├── general_PPLCNet_x2_5_lite_v1.0_infer.tar
    ├── picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer
    │   ├── infer_cfg.yml
    │   ├── inference.pdiparams
    │   ├── inference.pdiparams.info
    │   └── inference.pdmodel
    └── picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
    
    2 directories, 9 files


- 这里串联主体检测、特征提取、向量检索，从而构成一整套图像识别系统：

若商品为原索引库里已有的商品：
建立索引库


```python
# 建立索引库
%cd /home/aistudio/PaddleClas/deploy
!python3 python/build_gallery.py \
    -c configs/build_general.yaml \
    -o IndexProcess.data_file="/home/aistudio/dataset/data_file.txt" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index_inference"
```

### 3.抛物识别检索
识别图片 运行下面的命令，对图像 检测出的图形进行识别与检索并报警:


```python
#基于索引库的图像识别
%cd /home/aistudio/PaddleClas/deploy
!python python/predict_system.py \
    -c configs/inference_general.yaml \
    -o Global.infer_imgs="/home/aistudio/dataset/ex_obj.jpg" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index_inference"
```
