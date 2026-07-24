## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.045187955


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0139796, 0.0014696, -0.0139796, 0.0014696, -0.0154492, 0.0154492)
1: (-0.0144612, -0.0004636, -0.0144612, -0.0004636, -0.0139976, 0.0139976)
2: (0.0260131, 0.0689392, 0.0260131, 0.0689392, -0.0429261, 0.0429261)
3: (-0.0040237, 0.0273463, -0.0040237, 0.0273463, -0.0213450, 0.0213450)
4: (-0.0078113, 0.0021664, -0.0078113, 0.0021664, -0.0099778, 0.0099778)
5: (0.0085833, 0.0162496, 0.0085833, 0.0162496, -0.0076663, 0.0076663)
6: (-0.0309397, 0.0069420, -0.0309397, 0.0069420, -0.0366333, 0.0366333)
7: (0.9229466, 0.9828875, 0.9229466, 0.9828875, -0.0599409, 0.0599409)
8: (-0.0139797, 0.0183425, -0.0139797, 0.0183425, -0.0276768, 0.0276768)
9: (-0.0178200, 0.0093754, -0.0178200, 0.0093754, -0.0271954, 0.0271954)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 1.59 = 3.02 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0324880, upper bound: 0.0324880
