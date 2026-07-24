## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00017731


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0017372, -0.0013861, -0.0017372, -0.0013861, -0.0001805, 0.0001805)
1: (-0.0087190, -0.0078281, -0.0087190, -0.0078281, -0.0004579, 0.0004579)
2: (0.0296207, 0.0301735, 0.0296207, 0.0301735, -0.0002841, 0.0002841)
3: (0.0032210, 0.0042532, 0.0032210, 0.0042532, -0.0005305, 0.0005305)
4: (-0.0077618, -0.0068555, -0.0077618, -0.0068555, -0.0004658, 0.0004658)
5: (0.0107982, 0.0111415, 0.0107982, 0.0111415, -0.0001764, 0.0001764)
6: (0.0044760, 0.0057860, 0.0044760, 0.0057860, -0.0006733, 0.0006733)
7: (0.9811913, 0.9821080, 0.9811913, 0.9821080, -0.0004711, 0.0004711)
8: (-0.0067300, -0.0057473, -0.0067300, -0.0057473, -0.0005051, 0.0005051)
9: (-0.0012032, -0.0005540, -0.0012032, -0.0005540, -0.0003337, 0.0003337)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 1.26 = 2.68 seconds
status: Status.ADV_EXAMPLE
