## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00020622


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040944, -0.0040695, -0.0040944, -0.0040695, -0.0000140, 0.0000140)
1: (-0.0059903, -0.0050570, -0.0059903, -0.0050570, -0.0005254, 0.0005254)
2: (0.9692748, 0.9703948, 0.9692748, 0.9703948, -0.0006304, 0.0006304)
3: (0.0196824, 0.0279429, 0.0196824, 0.0279429, -0.0046501, 0.0046501)
4: (-0.0028183, -0.0021900, -0.0028183, -0.0021900, -0.0003537, 0.0003537)
5: (0.0144220, 0.0150570, 0.0144220, 0.0150570, -0.0003574, 0.0003574)
6: (0.0045831, 0.0048919, 0.0045831, 0.0048919, -0.0001739, 0.0001739)
7: (-0.0150199, -0.0128791, -0.0150199, -0.0128791, -0.0012051, 0.0012051)
8: (0.0048131, 0.0065115, 0.0048131, 0.0065115, -0.0009561, 0.0009561)
9: (0.0063814, 0.0094361, 0.0063814, 0.0094361, -0.0017196, 0.0017196)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.79 + 1.27 = 3.06 seconds
status: Status.ADV_EXAMPLE
