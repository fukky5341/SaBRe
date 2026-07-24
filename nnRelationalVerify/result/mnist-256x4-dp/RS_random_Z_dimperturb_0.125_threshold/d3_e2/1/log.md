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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0011211, 0.0021901, 0.0011211, 0.0021901, -0.0006417, 0.0006417)
1: (0.0014843, 0.0016387, 0.0014843, 0.0016387, -0.0000927, 0.0000927)
2: (0.0131490, 0.0137400, 0.0131490, 0.0137400, -0.0003548, 0.0003548)
3: (-0.0010811, -0.0004699, -0.0010811, -0.0004699, -0.0003670, 0.0003670)
4: (-0.0035283, -0.0028666, -0.0035283, -0.0028666, -0.0003973, 0.0003973)
5: (0.0068261, 0.0074523, 0.0068261, 0.0074523, -0.0003759, 0.0003759)
6: (0.0047835, 0.0072683, 0.0047835, 0.0072683, -0.0014916, 0.0014916)
7: (-0.0124555, -0.0090715, -0.0124555, -0.0090715, -0.0020314, 0.0020314)
8: (0.9804400, 0.9828237, 0.9804400, 0.9828237, -0.0014310, 0.0014310)
9: (-0.0002958, 0.0018680, -0.0002958, 0.0018680, -0.0012989, 0.0012989)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.63 + 1.20 = 2.83 seconds
status: Status.ADV_EXAMPLE
