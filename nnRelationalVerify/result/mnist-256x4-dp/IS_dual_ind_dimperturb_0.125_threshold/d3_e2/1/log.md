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
Threshold: 0.00088668


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0010950, 0.0022952, 0.0010950, 0.0022952, -0.0006995, 0.0006995)
1: (0.0014805, 0.0016539, 0.0014805, 0.0016539, -0.0001011, 0.0001011)
2: (0.0130909, 0.0137545, 0.0130909, 0.0137545, -0.0003868, 0.0003868)
3: (-0.0011412, -0.0004549, -0.0011412, -0.0004549, -0.0004000, 0.0004000)
4: (-0.0035445, -0.0028015, -0.0035445, -0.0028015, -0.0004330, 0.0004330)
5: (0.0067645, 0.0074676, 0.0067645, 0.0074676, -0.0004098, 0.0004098)
6: (0.0045393, 0.0073289, 0.0045393, 0.0073289, -0.0016259, 0.0016259)
7: (-0.0125380, -0.0087388, -0.0125380, -0.0087388, -0.0022143, 0.0022143)
8: (0.9803818, 0.9830581, 0.9803818, 0.9830581, -0.0015598, 0.0015598)
9: (-0.0005085, 0.0019208, -0.0005085, 0.0019208, -0.0014159, 0.0014159)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 1.22 = 2.87 seconds
status: Status.ADV_EXAMPLE
