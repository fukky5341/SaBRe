## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00147475


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0004321, 0.0030267, 0.0004321, 0.0030267, -0.0015369, 0.0015369)
1: (0.0013847, 0.0017596, 0.0013847, 0.0017596, -0.0002220, 0.0002220)
2: (0.0126865, 0.0141209, 0.0126865, 0.0141209, -0.0008497, 0.0008497)
3: (-0.0015595, -0.0000759, -0.0015595, -0.0000759, -0.0008788, 0.0008788)
4: (-0.0039548, -0.0023487, -0.0039548, -0.0023487, -0.0009514, 0.0009514)
5: (0.0063360, 0.0078559, 0.0063360, 0.0078559, -0.0009003, 0.0009003)
6: (0.0028392, 0.0088696, 0.0028392, 0.0088696, -0.0035722, 0.0035722)
7: (-0.0146363, -0.0064234, -0.0146363, -0.0064234, -0.0048651, 0.0048651)
8: (0.9789038, 0.9846890, 0.9789038, 0.9846890, -0.0034271, 0.0034271)
9: (-0.0019890, 0.0032625, -0.0019890, 0.0032625, -0.0031109, 0.0031109)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.28 = 2.82 seconds
status: Status.ADV_EXAMPLE
