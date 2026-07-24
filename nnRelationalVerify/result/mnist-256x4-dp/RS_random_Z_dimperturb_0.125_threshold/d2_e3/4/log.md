## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00100386


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021604, -0.0015286, -0.0021604, -0.0015286, -0.0002868, 0.0002868)
1: (-0.0023236, -0.0004465, -0.0023236, -0.0004465, -0.0008868, 0.0008868)
2: (0.0046765, 0.0062989, 0.0046765, 0.0062989, -0.0007778, 0.0007778)
3: (-0.0041665, -0.0039946, -0.0041665, -0.0039946, -0.0000800, 0.0000800)
4: (0.0044851, 0.0058014, 0.0044851, 0.0058014, -0.0006489, 0.0006489)
5: (-0.0009996, 0.0003675, -0.0009996, 0.0003675, -0.0006870, 0.0006870)
6: (-0.0055642, -0.0048463, -0.0055642, -0.0048463, -0.0003508, 0.0003508)
7: (0.0007822, 0.0020580, 0.0007822, 0.0020580, -0.0005970, 0.0005970)
8: (-0.0004157, -0.0002415, -0.0004157, -0.0002415, -0.0000938, 0.0000938)
9: (1.0046695, 1.0080512, 1.0046695, 1.0080512, -0.0016639, 0.0016639)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 1.24 = 2.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0010702, upper bound: 0.0010702

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0010254, upper bound: 0.0008364
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0008364, upper bound: 0.0010255
time: 0.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.87 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 9, lower bound: -0.0010254, upper bound: 0.0008364
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 9, lower bound: -0.0008364, upper bound: 0.0010255

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021604, -0.0015286, -0.0021604, -0.0015286, -0.0002139, 0.0002055
1: -0.0023236, -0.0004465, -0.0023236, -0.0004465, -0.0005356, 0.0005772
2: 0.0046765, 0.0062989, 0.0046765, 0.0062989, -0.0004508, 0.0005035
3: -0.0041665, -0.0039946, -0.0041665, -0.0039946, -0.0000686, 0.0000673
4: 0.0044851, 0.0058014, 0.0044851, 0.0058014, -0.0004158, 0.0003766
5: -0.0009996, 0.0003675, -0.0009996, 0.0003675, -0.0003707, 0.0004176
6: -0.0055642, -0.0048463, -0.0055642, -0.0048463, -0.0002202, 0.0002337
7: 0.0007822, 0.0020580, 0.0007822, 0.0020580, -0.0004288, 0.0003973
8: -0.0004157, -0.0002415, -0.0004157, -0.0002415, -0.0000584, 0.0000641
9: 1.0046695, 1.0080512, 1.0046695, 1.0080512, -0.0010371, 0.0009200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0009919, upper bound: 0.0007839
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0009875, upper bound: 0.0007957
time: 0.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021604, -0.0015286, -0.0021604, -0.0015286, -0.0002055, 0.0002139
1: -0.0023236, -0.0004465, -0.0023236, -0.0004465, -0.0005772, 0.0005356
2: 0.0046765, 0.0062989, 0.0046765, 0.0062989, -0.0005035, 0.0004508
3: -0.0041665, -0.0039946, -0.0041665, -0.0039946, -0.0000673, 0.0000686
4: 0.0044851, 0.0058014, 0.0044851, 0.0058014, -0.0003766, 0.0004158
5: -0.0009996, 0.0003675, -0.0009996, 0.0003675, -0.0004176, 0.0003707
6: -0.0055642, -0.0048463, -0.0055642, -0.0048463, -0.0002337, 0.0002202
7: 0.0007822, 0.0020580, 0.0007822, 0.0020580, -0.0003973, 0.0004288
8: -0.0004157, -0.0002415, -0.0004157, -0.0002415, -0.0000641, 0.0000584
9: 1.0046695, 1.0080512, 1.0046695, 1.0080512, -0.0009200, 0.0010371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0007957, upper bound: 0.0009875
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0007839, upper bound: 0.0009919
time: 0.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 9, lower bound: -0.0009919, upper bound: 0.0007839
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 9, lower bound: -0.0009875, upper bound: 0.0007957
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 9, lower bound: -0.0007957, upper bound: 0.0009875
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 9, lower bound: -0.0007839, upper bound: 0.0009919

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.62 + 5.45 = 8.06 seconds
