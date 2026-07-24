## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00026658


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9881809, 0.9887577, 0.9881809, 0.9887577, -0.0004408, 0.0004408)
1: (-0.0042090, -0.0040652, -0.0042090, -0.0040652, -0.0001098, 0.0001098)
2: (0.0114895, 0.0122513, 0.0114895, 0.0122513, -0.0005821, 0.0005821)
3: (-0.0068494, -0.0065027, -0.0068494, -0.0065027, -0.0002650, 0.0002650)
4: (0.0027517, 0.0028991, 0.0027517, 0.0028991, -0.0001127, 0.0001127)
5: (0.0134103, 0.0143683, 0.0134103, 0.0143683, -0.0007321, 0.0007321)
6: (-0.0021060, -0.0018628, -0.0021060, -0.0018628, -0.0001858, 0.0001858)
7: (-0.0085865, -0.0079574, -0.0085865, -0.0079574, -0.0004808, 0.0004808)
8: (-0.0040797, -0.0037488, -0.0040797, -0.0037488, -0.0002528, 0.0002528)
9: (0.0024831, 0.0028668, 0.0024831, 0.0028668, -0.0002932, 0.0002932)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.28 = 2.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0002868, upper bound: 0.0002868

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002605, upper bound: 0.0002431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002431, upper bound: 0.0002605
time: 0.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.90 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.90
Output dim: 0, lower bound: -0.0002605, upper bound: 0.0002431
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.90
Output dim: 0, lower bound: -0.0002431, upper bound: 0.0002605

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.59 + 0.90 = 3.49 seconds
