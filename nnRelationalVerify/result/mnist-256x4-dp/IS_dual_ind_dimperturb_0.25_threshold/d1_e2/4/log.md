## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00104286


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041986, -0.0041727, -0.0041986, -0.0041727, -0.0000258, 0.0000258)
1: (-0.0098921, -0.0095082, -0.0098921, -0.0095082, -0.0003839, 0.0003839)
2: (0.9645926, 0.9650532, 0.9645926, 0.9650532, -0.0004606, 0.0004606)
3: (-0.0148532, -0.0114554, -0.0148532, -0.0114554, -0.0025350, 0.0025350)
4: (0.0001782, 0.0004366, 0.0001782, 0.0004366, -0.0002584, 0.0002584)
5: (0.0174505, 0.0178039, 0.0174505, 0.0178039, -0.0003535, 0.0003535)
6: (0.0030461, 0.0034189, 0.0030461, 0.0034189, -0.0003728, 0.0003728)
7: (-0.0048095, -0.0037830, -0.0048095, -0.0037830, -0.0010265, 0.0010265)
8: (0.0129135, 0.0136121, 0.0129135, 0.0136121, -0.0006986, 0.0006986)
9: (0.0209508, 0.0222073, 0.0209508, 0.0222073, -0.0011613, 0.0011613)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.20 + 1.32 = 2.52 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0002735, upper bound: 0.0002735
