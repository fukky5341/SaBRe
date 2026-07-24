## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00527912


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0008390, 0.0014462, -0.0008390, 0.0014462, -0.0022852, 0.0022852)
1: (-0.0034389, -0.0025173, -0.0034389, -0.0025173, -0.0009216, 0.0009216)
2: (0.0323349, 0.0338195, 0.0323349, 0.0338195, -0.0014846, 0.0014846)
3: (-0.0031791, -0.0014674, -0.0031791, -0.0014674, -0.0017116, 0.0017116)
4: (-0.0023013, -0.0010352, -0.0023013, -0.0010352, -0.0011014, 0.0011014)
5: (0.0117412, 0.0138702, 0.0117412, 0.0138702, -0.0021290, 0.0021290)
6: (-0.0036722, -0.0022720, -0.0036722, -0.0022720, -0.0011629, 0.0011629)
7: (0.9757581, 0.9766271, 0.9757581, 0.9766271, -0.0008690, 0.0008690)
8: (-0.0144819, -0.0085430, -0.0144819, -0.0085430, -0.0059389, 0.0059389)
9: (0.0009421, 0.0043798, 0.0009421, 0.0043798, -0.0034377, 0.0034377)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 1.30 = 2.51 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0004488, upper bound: 0.0004488
