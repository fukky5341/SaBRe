## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0125892


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0005426, 0.0007787, -0.0005426, 0.0007787, -0.0011897, 0.0011897)
1: (-0.0012200, 0.0025150, -0.0012200, 0.0025150, -0.0036883, 0.0036883)
2: (0.0125735, 0.0181670, 0.0125735, 0.0181670, -0.0053793, 0.0053793)
3: (-0.0011722, 0.0030339, -0.0011722, 0.0030339, -0.0039817, 0.0039817)
4: (-0.0054608, -0.0015811, -0.0054608, -0.0015811, -0.0038797, 0.0038797)
5: (0.0067681, 0.0109666, 0.0067681, 0.0109666, -0.0039686, 0.0039686)
6: (0.0080641, 0.0103788, 0.0080641, 0.0103788, -0.0023147, 0.0023147)
7: (-0.0222066, -0.0130924, -0.0222066, -0.0130924, -0.0080235, 0.0080235)
8: (0.9601662, 0.9862796, 0.9601662, 0.9862796, -0.0252593, 0.0252593)
9: (0.0016855, 0.0093603, 0.0016855, 0.0093603, -0.0069187, 0.0069187)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.98 = 3.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0168792, upper bound: 0.0168792

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0164741, upper bound: 0.0164741
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0164741, upper bound: 0.0164741
time: 0.94 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.99
Output dim: 8, lower bound: -0.0164741, upper bound: 0.0164741
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.99
Output dim: 8, lower bound: -0.0164741, upper bound: 0.0164741

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005426, 0.0007787, -0.0005426, 0.0007787, -0.0011874, 0.0011876
1: -0.0012200, 0.0025150, -0.0012200, 0.0025150, -0.0036783, 0.0036792
2: 0.0125735, 0.0181670, 0.0125735, 0.0181670, -0.0053655, 0.0053642
3: -0.0011722, 0.0030339, -0.0011722, 0.0030339, -0.0039713, 0.0039704
4: -0.0054608, -0.0015811, -0.0054608, -0.0015811, -0.0038797, 0.0038797
5: 0.0067681, 0.0109666, 0.0067681, 0.0109666, -0.0039582, 0.0039573
6: 0.0080641, 0.0103788, 0.0080641, 0.0103788, -0.0023147, 0.0023147
7: -0.0222066, -0.0130924, -0.0222066, -0.0130924, -0.0079990, 0.0080011
8: 0.9601662, 0.9862796, 0.9601662, 0.9862796, -0.0251892, 0.0251951
9: 0.0016855, 0.0093603, 0.0016855, 0.0093603, -0.0068998, 0.0068981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119863, upper bound: 0.0119863
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119863, upper bound: 0.0119863
time: 0.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005426, 0.0007787, -0.0005426, 0.0007787, -0.0011876, 0.0011897
1: -0.0012200, 0.0025150, -0.0012200, 0.0025150, -0.0036792, 0.0036883
2: 0.0125735, 0.0181670, 0.0125735, 0.0181670, -0.0053793, 0.0053655
3: -0.0011722, 0.0030339, -0.0011722, 0.0030339, -0.0039817, 0.0039713
4: -0.0054608, -0.0015811, -0.0054608, -0.0015811, -0.0038797, 0.0038797
5: 0.0067681, 0.0109666, 0.0067681, 0.0109666, -0.0039686, 0.0039583
6: 0.0080641, 0.0103788, 0.0080641, 0.0103788, -0.0023147, 0.0023147
7: -0.0222066, -0.0130924, -0.0222066, -0.0130924, -0.0080011, 0.0080235
8: 0.9601662, 0.9862796, 0.9601662, 0.9862796, -0.0251951, 0.0252593
9: 0.0016855, 0.0093603, 0.0016855, 0.0093603, -0.0069187, 0.0068998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119863, upper bound: 0.0119863
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119863, upper bound: 0.0119863
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 8, lower bound: -0.0119863, upper bound: 0.0119863
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 8, lower bound: -0.0119863, upper bound: 0.0119863
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 8, lower bound: -0.0119863, upper bound: 0.0119863
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 8, lower bound: -0.0119863, upper bound: 0.0119863

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.24 + 6.53 = 9.78 seconds
