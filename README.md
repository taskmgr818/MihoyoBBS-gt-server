# geetest-v3-click-server
极验三代九宫格、图标点选的打码服务端

**本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

**本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

**本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

# 使用方法

## 服务端
* [训练模型](https://github.com/taskmgr818/MihoyoBBS-gt-server/blob/main/train.md)

* 将resnet18.onnx、siamese.onnx分别放入predict/nine、predict/icon

* 安装依赖

```commandline
pip install -r requirements.txt
```
* 运行

```commandline
python3 main.py
```

## 调用接口

``` python
import httpx

def geetest(gt, challenge):
    data = httpx.post(
        "http://127.0.0.1:10721",
        json={"gt": gt, "challenge": challenge},
        timeout=30,
    ).json()
    if data["status"] == "success":
        return data["validate"]
    return None
```

成功率接近100%（1000次测试全部成功）

# 实现思路

## 九宫格
* 采用均方误差（MSE）法对小图标进行分类
* 使用resnet18对图片进行分类

## 图标点选
* 使用ddddocr进行目标检测
* 使用孪生神经网络计算相似度

# 协议
本项目遵循 AGPL-3.0 协议开源，请遵守相关协议。

# 鸣谢
[ravizhan/geetest-v3-click-crack](https://github.com/ravizhan/geetest-v3-click-crack) 提供极验接口逆向

[taisuii/ClassificationCaptchaOcr](https://github.com/taisuii/ClassificationCaptchaOcr) 提供九宫格识别思路及模型

[bubbliiiing/Siamese-pytorch](https://github.com/bubbliiiing/Siamese-pytorch) 提供孪生网络模型

[sml2h3/ddddocr](https://github.com/sml2h3/ddddocr) 提供目标检测SDK