# 训练模型
## resnet18
参见 [taisuii/ClassificationCaptchaOcr](https://github.com/taisuii/ClassificationCaptchaOcr)
## siamese
参见 [bubbliiiing/Siamese-pytorch](https://github.com/bubbliiiing/Siamese-pytorch)
* 每个分类图片数不得低于三张
* 使用以下代码导出
``` python
import torch
from nets.siamese import Siamese

weights = "logs/last_epoch_weights.pth"  # 选择合适的权重
onnx_outpath = "siamese.onnx"

model = Siamese([105, 105])
model.load_state_dict(torch.load(weights))
input_image = [torch.randn(1, 3, 105, 105), torch.randn(1, 3, 105, 105)]
torch.onnx.export(
    model,
    input_image,
    onnx_outpath,
    opset_version=13,
    verbose=True,
    do_constant_folding=True,
    input_names=["input", "input.53"],
    output_names=["output"],
)
```