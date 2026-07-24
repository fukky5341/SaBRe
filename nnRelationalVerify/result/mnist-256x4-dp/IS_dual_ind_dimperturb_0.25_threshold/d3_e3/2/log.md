## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.002363328


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0001992, 0.0011274, 0.0001992, 0.0011274, -0.0009281, 0.0009281)
1: (0.9932793, 0.9959592, 0.9932793, 0.9959592, -0.0026799, 0.0026799)
2: (-0.0101437, -0.0017468, -0.0101437, -0.0017468, -0.0078358, 0.0078358)
3: (0.0030197, 0.0042512, 0.0030197, 0.0042512, -0.0012315, 0.0012315)
4: (-0.0002024, 0.0064341, -0.0002024, 0.0064341, -0.0066365, 0.0066365)
5: (0.0041390, 0.0072187, 0.0041390, 0.0072187, -0.0030797, 0.0030797)
6: (-0.0026421, 0.0002726, -0.0026421, 0.0002726, -0.0029147, 0.0029147)
7: (-0.0089412, -0.0066214, -0.0089412, -0.0066214, -0.0023197, 0.0023197)
8: (0.0005467, 0.0115793, 0.0005467, 0.0115793, -0.0108823, 0.0108823)
9: (-0.0037212, -0.0008191, -0.0037212, -0.0008191, -0.0029021, 0.0029021)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.75 + 2.08 = 3.82 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0022586, upper bound: 0.0022588
