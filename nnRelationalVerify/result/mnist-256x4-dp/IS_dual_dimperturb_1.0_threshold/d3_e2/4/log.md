## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.027818040465442097


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693)
1: (-0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900)
2: (0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209)
3: (-0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987)
4: (0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823)
5: (-0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120)
6: (-0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657)
7: (-0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077)
8: (-0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964)
9: (-0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 3.51 = 4.92 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -0.0231797, upper bound: 0.0231797
