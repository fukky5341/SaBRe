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
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512)
1: (-0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543)
2: (0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545)
3: (-0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0449331, 0.0449332)
4: (-0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369)
5: (0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232)
6: (-0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519)
7: (0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155)
8: (-0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0444801, 0.0444801)
9: (-0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.35 + 1.84 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0562686, upper bound: 0.0562686

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0555126, upper bound: 0.0557300
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0557300, upper bound: 0.0555126
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 7, lower bound: -0.0555126, upper bound: 0.0557300
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 7, lower bound: -0.0557300, upper bound: 0.0555126

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0447842, 0.0447503
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0442951, 0.0442511
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0538941, upper bound: 0.0538658
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0538479, upper bound: 0.0539243
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0447503, 0.0447842
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0442511, 0.0442951
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0539243, upper bound: 0.0538479
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0538658, upper bound: 0.0538941
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 7, lower bound: -0.0538941, upper bound: 0.0538658
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 7, lower bound: -0.0538479, upper bound: 0.0539243
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 7, lower bound: -0.0539243, upper bound: 0.0538479
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 7, lower bound: -0.0538658, upper bound: 0.0538941

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0446023, 0.0446488
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0440774, 0.0441376
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0538784, upper bound: 0.0537976
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0538273, upper bound: 0.0538498
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0446823, 0.0445685
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0441811, 0.0440334
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0538319, upper bound: 0.0538511
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0537726, upper bound: 0.0539084
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0445685, 0.0446822
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0440334, 0.0441811
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0539084, upper bound: 0.0537726
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0538511, upper bound: 0.0538319
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0446488, 0.0446023
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0441376, 0.0440774
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0538498, upper bound: 0.0538273
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0537976, upper bound: 0.0538784
time: 0.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0538784, upper bound: 0.0537976
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0538273, upper bound: 0.0538498
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0538319, upper bound: 0.0538511
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0537726, upper bound: 0.0539084
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0539084, upper bound: 0.0537726
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0538511, upper bound: 0.0538319
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0538498, upper bound: 0.0538273
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0537976, upper bound: 0.0538784

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0445945, 0.0446458
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0440201, 0.0440867
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0535225, upper bound: 0.0528944
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0528813, upper bound: 0.0534294
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0445994, 0.0446375
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0440265, 0.0440760
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0534674, upper bound: 0.0529418
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0528321, upper bound: 0.0534848
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0446721, 0.0445655
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0441209, 0.0439825
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0534672, upper bound: 0.0529000
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0528699, upper bound: 0.0534918
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0446792, 0.0445606
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0441302, 0.0439761
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0534032, upper bound: 0.0529523
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0528192, upper bound: 0.0535459
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0445606, 0.0446792
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0439761, 0.0441302
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0535459, upper bound: 0.0528192
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0529523, upper bound: 0.0534032
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0445654, 0.0446721
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0439825, 0.0441209
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0534918, upper bound: 0.0528699
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0529000, upper bound: 0.0534672
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0446375, 0.0445993
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0440760, 0.0440265
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0534848, upper bound: 0.0528321
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0529418, upper bound: 0.0534674
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0446458, 0.0445945
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0440867, 0.0440201
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0534294, upper bound: 0.0528813
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0528944, upper bound: 0.0535225
time: 0.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0535225, upper bound: 0.0528944
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0528813, upper bound: 0.0534294
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0534674, upper bound: 0.0529418
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0528321, upper bound: 0.0534848
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0534672, upper bound: 0.0529000
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0528699, upper bound: 0.0534918
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0534032, upper bound: 0.0529523
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0528192, upper bound: 0.0535459
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0535459, upper bound: 0.0528192
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0529523, upper bound: 0.0534032
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0534918, upper bound: 0.0528699
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0529000, upper bound: 0.0534672
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0534848, upper bound: 0.0528321
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0529418, upper bound: 0.0534674
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0534294, upper bound: 0.0528813
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.0528944, upper bound: 0.0535225

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435092, 0.0437547
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426585, 0.0429772
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0525548, upper bound: 0.0527245
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0533766, upper bound: 0.0505538
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436929, 0.0435605
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0428969, 0.0427251
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0511248, upper bound: 0.0532729
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0527131, upper bound: 0.0521696
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435141, 0.0437495
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426648, 0.0429704
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0525264, upper bound: 0.0527683
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0533206, upper bound: 0.0505516
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436969, 0.0435522
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429022, 0.0427144
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0511156, upper bound: 0.0533294
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0526628, upper bound: 0.0522117
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435869, 0.0436704
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0427593, 0.0428678
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0522117, upper bound: 0.0527389
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0533059, upper bound: 0.0511511
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437785, 0.0434802
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0430081, 0.0426208
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0504921, upper bound: 0.0533459
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0526912, upper bound: 0.0525308
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435940, 0.0436647
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0427686, 0.0428604
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0521677, upper bound: 0.0527889
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532423, upper bound: 0.0511604
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437825, 0.0434753
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0430132, 0.0426145
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0504910, upper bound: 0.0533991
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0526394, upper bound: 0.0525590
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0434753, 0.0437825
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426145, 0.0430132
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0525590, upper bound: 0.0526394
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0533991, upper bound: 0.0504910
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436647, 0.0435940
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0428604, 0.0427685
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0511604, upper bound: 0.0532423
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0527889, upper bound: 0.0521677
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0434802, 0.0437785
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426208, 0.0430081
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0525308, upper bound: 0.0526912
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0533459, upper bound: 0.0504921
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436704, 0.0435869
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0428678, 0.0427593
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0511511, upper bound: 0.0533059
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0527389, upper bound: 0.0522117
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435522, 0.0436969
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0427144, 0.0429022
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0522117, upper bound: 0.0526628
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0533294, upper bound: 0.0511156
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437495, 0.0435141
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429704, 0.0426648
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0505516, upper bound: 0.0533206
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0527683, upper bound: 0.0525264
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435605, 0.0436929
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0427251, 0.0428969
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0521696, upper bound: 0.0527131
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532729, upper bound: 0.0511248
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437547, 0.0435092
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429772, 0.0426585
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 2.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0505538, upper bound: 0.0533766
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0527245, upper bound: 0.0525548
time: 0.81 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0525548, upper bound: 0.0527245
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0533766, upper bound: 0.0505538
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0511248, upper bound: 0.0532729
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0527131, upper bound: 0.0521696
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0525264, upper bound: 0.0527683
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0533206, upper bound: 0.0505516
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0511156, upper bound: 0.0533294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0526628, upper bound: 0.0522117
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0522117, upper bound: 0.0527389
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0533059, upper bound: 0.0511511
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0504921, upper bound: 0.0533459
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0526912, upper bound: 0.0525308
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0521677, upper bound: 0.0527889
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0532423, upper bound: 0.0511604
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0504910, upper bound: 0.0533991
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0526394, upper bound: 0.0525590
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0525590, upper bound: 0.0526394
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0533991, upper bound: 0.0504910
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0511604, upper bound: 0.0532423
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0527889, upper bound: 0.0521677
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0525308, upper bound: 0.0526912
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0533459, upper bound: 0.0504921
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0511511, upper bound: 0.0533059
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0527389, upper bound: 0.0522117
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0522117, upper bound: 0.0526628
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0533294, upper bound: 0.0511156
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0505516, upper bound: 0.0533206
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0527683, upper bound: 0.0525264
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0521696, upper bound: 0.0527131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0532729, upper bound: 0.0511248
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0505538, upper bound: 0.0533766
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.02
Output dim: 7, lower bound: -0.0527245, upper bound: 0.0525548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435604, 0.0437791
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426273, 0.0429116
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0508776, upper bound: 0.0518759
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0517278, upper bound: 0.0513429
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435183, 0.0438059
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0425729, 0.0429459
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518474, upper bound: 0.0498180
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0525378, upper bound: 0.0493221
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437440, 0.0435760
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0428657, 0.0426478
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0496876, upper bound: 0.0524287
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0503398, upper bound: 0.0518503
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437124, 0.0436116
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0428250, 0.0426938
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513129, upper bound: 0.0513694
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518982, upper bound: 0.0506399
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435652, 0.0437744
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426336, 0.0429054
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0508704, upper bound: 0.0519456
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0516890, upper bound: 0.0513451
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435223, 0.0438007
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0425781, 0.0429392
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518350, upper bound: 0.0498180
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0524669, upper bound: 0.0493220
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437481, 0.0435691
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0428709, 0.0426389
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0496882, upper bound: 0.0525018
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0503202, upper bound: 0.0518586
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437163, 0.0436034
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0428300, 0.0426831
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513107, upper bound: 0.0514254
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518246, upper bound: 0.0506529
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436380, 0.0436914
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0427281, 0.0427977
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0506495, upper bound: 0.0518817
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0514254, upper bound: 0.0513437
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436029, 0.0437216
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426828, 0.0428366
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518408, upper bound: 0.0503490
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0524889, upper bound: 0.0497337
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0438297, 0.0434903
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429769, 0.0425366
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0492607, upper bound: 0.0524899
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0497713, upper bound: 0.0518553
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0438033, 0.0435313
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429429, 0.0425896
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513058, upper bound: 0.0516955
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518935, upper bound: 0.0508739
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436451, 0.0436861
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0427373, 0.0427907
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0506330, upper bound: 0.0519528
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513694, upper bound: 0.0513457
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436087, 0.0437159
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426903, 0.0428291
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518238, upper bound: 0.0503584
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0524097, upper bound: 0.0497333
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0438336, 0.0434849
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429820, 0.0425296
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0492545, upper bound: 0.0525559
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0497696, upper bound: 0.0518633
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0438061, 0.0435264
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429466, 0.0425832
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513013, upper bound: 0.0517345
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518144, upper bound: 0.0508820
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435264, 0.0438062
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0425832, 0.0429466
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0508820, upper bound: 0.0518144
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0517345, upper bound: 0.0513013
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0434849, 0.0438336
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0425296, 0.0429820
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518633, upper bound: 0.0497696
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0525559, upper bound: 0.0492545
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437159, 0.0436087
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0428291, 0.0426903
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0497333, upper bound: 0.0524097
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0503584, upper bound: 0.0518238
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436861, 0.0436451
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0427908, 0.0427373
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513457, upper bound: 0.0513694
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0519528, upper bound: 0.0506330
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435313, 0.0438033
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0425896, 0.0429429
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0508739, upper bound: 0.0518935
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0516955, upper bound: 0.0513058
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0434903, 0.0438297
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0425366, 0.0429769
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518553, upper bound: 0.0497713
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0524899, upper bound: 0.0492607
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437216, 0.0436029
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0428366, 0.0426828
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0497337, upper bound: 0.0524889
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0503490, upper bound: 0.0518408
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436914, 0.0436380
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0427977, 0.0427280
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513437, upper bound: 0.0514254
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518817, upper bound: 0.0506495
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436034, 0.0437163
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426831, 0.0428300
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0506529, upper bound: 0.0518246
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0514254, upper bound: 0.0513107
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435691, 0.0437481
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426389, 0.0428709
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518586, upper bound: 0.0503202
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0525018, upper bound: 0.0496882
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0438007, 0.0435223
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429392, 0.0425781
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0493220, upper bound: 0.0524669
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0498180, upper bound: 0.0518350
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437744, 0.0435652
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429054, 0.0426336
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513451, upper bound: 0.0516890
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0519456, upper bound: 0.0508704
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0436116, 0.0437124
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426938, 0.0428250
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0506399, upper bound: 0.0518982
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513694, upper bound: 0.0513129
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0435760, 0.0437440
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0426479, 0.0428657
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518503, upper bound: 0.0503398
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0524287, upper bound: 0.0496876
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0438058, 0.0435183
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429459, 0.0425729
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0493221, upper bound: 0.0525378
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0498180, upper bound: 0.0518474
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0437792, 0.0435604
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0429116, 0.0426273
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0513429, upper bound: 0.0517278
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0518759, upper bound: 0.0508776
time: 0.83 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0508776, upper bound: 0.0518759
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0517278, upper bound: 0.0513429
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518474, upper bound: 0.0498180
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0525378, upper bound: 0.0493221
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0496876, upper bound: 0.0524287
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0503398, upper bound: 0.0518503
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513129, upper bound: 0.0513694
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518982, upper bound: 0.0506399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0508704, upper bound: 0.0519456
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0516890, upper bound: 0.0513451
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518350, upper bound: 0.0498180
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0524669, upper bound: 0.0493220
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0496882, upper bound: 0.0525018
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0503202, upper bound: 0.0518586
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513107, upper bound: 0.0514254
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518246, upper bound: 0.0506529
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0506495, upper bound: 0.0518817
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0514254, upper bound: 0.0513437
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518408, upper bound: 0.0503490
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0524889, upper bound: 0.0497337
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0492607, upper bound: 0.0524899
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0497713, upper bound: 0.0518553
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513058, upper bound: 0.0516955
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518935, upper bound: 0.0508739
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0506330, upper bound: 0.0519528
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513694, upper bound: 0.0513457
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518238, upper bound: 0.0503584
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0524097, upper bound: 0.0497333
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0492545, upper bound: 0.0525559
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0497696, upper bound: 0.0518633
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513013, upper bound: 0.0517345
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518144, upper bound: 0.0508820
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0508820, upper bound: 0.0518144
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0517345, upper bound: 0.0513013
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518633, upper bound: 0.0497696
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0525559, upper bound: 0.0492545
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0497333, upper bound: 0.0524097
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0503584, upper bound: 0.0518238
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513457, upper bound: 0.0513694
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0519528, upper bound: 0.0506330
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0508739, upper bound: 0.0518935
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0516955, upper bound: 0.0513058
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518553, upper bound: 0.0497713
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0524899, upper bound: 0.0492607
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0497337, upper bound: 0.0524889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0503490, upper bound: 0.0518408
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513437, upper bound: 0.0514254
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518817, upper bound: 0.0506495
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0506529, upper bound: 0.0518246
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0514254, upper bound: 0.0513107
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518586, upper bound: 0.0503202
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0525018, upper bound: 0.0496882
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0493220, upper bound: 0.0524669
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0498180, upper bound: 0.0518350
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513451, upper bound: 0.0516890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0519456, upper bound: 0.0508704
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0506399, upper bound: 0.0518982
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513694, upper bound: 0.0513129
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518503, upper bound: 0.0503398
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0524287, upper bound: 0.0496876
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0493221, upper bound: 0.0525378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0498180, upper bound: 0.0518474
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0513429, upper bound: 0.0517278
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 7, lower bound: -0.0518759, upper bound: 0.0508776

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425683, 0.0426718
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414950, 0.0416303
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0471320, upper bound: 0.0455026
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0459555, upper bound: 0.0476713
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0424263, 0.0428111
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0413117, 0.0418101
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0476532, upper bound: 0.0454909
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460226, upper bound: 0.0472948
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425683, 0.0426718
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414950, 0.0416303
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0478869, upper bound: 0.0449350
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0461406, upper bound: 0.0461039
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0424263, 0.0428111
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0413117, 0.0418101
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0483557, upper bound: 0.0448971
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0461652, upper bound: 0.0457847
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0427314, 0.0424776
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0417067, 0.0413782
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460202, upper bound: 0.0458064
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0453032, upper bound: 0.0481915
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426100, 0.0426334
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415501, 0.0415795
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0463928, upper bound: 0.0457991
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0454346, upper bound: 0.0478049
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0427314, 0.0424776
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0417067, 0.0413782
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0473615, upper bound: 0.0456007
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457932, upper bound: 0.0473339
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426100, 0.0426334
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415501, 0.0415795
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0477613, upper bound: 0.0455780
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0458230, upper bound: 0.0469074
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425768, 0.0426666
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415059, 0.0416236
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0470974, upper bound: 0.0457574
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457660, upper bound: 0.0477600
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0424312, 0.0428051
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0413180, 0.0418023
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0475848, upper bound: 0.0457455
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457939, upper bound: 0.0473664
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425768, 0.0426666
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415059, 0.0416236
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0478105, upper bound: 0.0450880
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0458892, upper bound: 0.0461142
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0424312, 0.0428051
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0413180, 0.0418023
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0482309, upper bound: 0.0450206
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0458986, upper bound: 0.0457962
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0427364, 0.0424694
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0417131, 0.0413675
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460022, upper bound: 0.0460774
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0451595, upper bound: 0.0483183
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426140, 0.0426221
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415553, 0.0415648
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0463626, upper bound: 0.0460625
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0452377, upper bound: 0.0478775
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0427364, 0.0424694
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0417131, 0.0413675
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0472892, upper bound: 0.0457984
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0455523, upper bound: 0.0474328
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426140, 0.0426221
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415553, 0.0415648
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0476675, upper bound: 0.0457496
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0455643, upper bound: 0.0469522
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426596, 0.0425876
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416134, 0.0415210
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0469522, upper bound: 0.0455646
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457496, upper bound: 0.0476777
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425040, 0.0427138
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414124, 0.0416838
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0474328, upper bound: 0.0455523
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457984, upper bound: 0.0472949
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426596, 0.0425876
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416134, 0.0415210
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0478774, upper bound: 0.0452380
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460625, upper bound: 0.0463626
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425040, 0.0427138
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414124, 0.0416838
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0483183, upper bound: 0.0451597
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460774, upper bound: 0.0460022
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0428315, 0.0423973
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0418367, 0.0412740
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457962, upper bound: 0.0458986
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0450206, upper bound: 0.0482315
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426956, 0.0425445
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416613, 0.0414640
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0461142, upper bound: 0.0458892
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0450880, upper bound: 0.0478105
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0428315, 0.0423973
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0418367, 0.0412740
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0473611, upper bound: 0.0457939
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457432, upper bound: 0.0475848
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426956, 0.0425445
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416613, 0.0414640
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0477576, upper bound: 0.0457660
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457570, upper bound: 0.0470974
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426672, 0.0425818
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416234, 0.0415135
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0469074, upper bound: 0.0458230
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0455780, upper bound: 0.0477653
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425111, 0.0427080
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414217, 0.0416762
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0473339, upper bound: 0.0457934
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0456007, upper bound: 0.0473664
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426672, 0.0425818
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416234, 0.0415135
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0478049, upper bound: 0.0454346
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457991, upper bound: 0.0463928
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425111, 0.0427080
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414217, 0.0416762
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0481915, upper bound: 0.0453032
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0458064, upper bound: 0.0460202
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0428376, 0.0423924
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0418445, 0.0412676
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457847, upper bound: 0.0461652
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0448971, upper bound: 0.0483558
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426996, 0.0425329
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416664, 0.0414489
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0461039, upper bound: 0.0461406
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0449350, upper bound: 0.0478869
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0428376, 0.0423924
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0418445, 0.0412676
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0472892, upper bound: 0.0460226
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0454887, upper bound: 0.0476532
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426996, 0.0425329
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416664, 0.0414489
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0476608, upper bound: 0.0459555
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0455019, upper bound: 0.0471320
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425329, 0.0426996
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414490, 0.0416664
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0471320, upper bound: 0.0455019
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0459555, upper bound: 0.0476608
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0423924, 0.0428376
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0412676, 0.0418446
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0476532, upper bound: 0.0454887
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460226, upper bound: 0.0472892
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425329, 0.0426996
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414490, 0.0416664
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0478869, upper bound: 0.0449350
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0461406, upper bound: 0.0461039
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0423924, 0.0428376
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0412676, 0.0418446
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0483558, upper bound: 0.0448971
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0461652, upper bound: 0.0457847
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0427080, 0.0425111
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416762, 0.0414217
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460202, upper bound: 0.0458064
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0453032, upper bound: 0.0481915
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425819, 0.0426672
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415135, 0.0416233
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0463928, upper bound: 0.0457991
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0454346, upper bound: 0.0478049
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0427080, 0.0425111
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416762, 0.0414217
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0473664, upper bound: 0.0456007
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457934, upper bound: 0.0473339
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425819, 0.0426672
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415135, 0.0416233
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0477653, upper bound: 0.0455780
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0458230, upper bound: 0.0469074
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425445, 0.0426956
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414640, 0.0416613
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0470974, upper bound: 0.0457570
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457660, upper bound: 0.0477576
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0423973, 0.0428315
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0412740, 0.0418367
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0475848, upper bound: 0.0457432
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457939, upper bound: 0.0473611
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425445, 0.0426956
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0414640, 0.0416613
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0478105, upper bound: 0.0450880
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0458892, upper bound: 0.0461142
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0423973, 0.0428315
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0412740, 0.0418367
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0482315, upper bound: 0.0450206
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0458986, upper bound: 0.0457962
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0427138, 0.0425040
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416839, 0.0414124
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460022, upper bound: 0.0460774
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0451597, upper bound: 0.0483183
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425876, 0.0426596
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415210, 0.0416134
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0463626, upper bound: 0.0460625
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0452380, upper bound: 0.0478774
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0427138, 0.0425040
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416839, 0.0414124
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0472949, upper bound: 0.0457984
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0455523, upper bound: 0.0474328
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0425876, 0.0426596
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415210, 0.0416134
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0476777, upper bound: 0.0457496
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0455646, upper bound: 0.0469522
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426221, 0.0426140
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415648, 0.0415553
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0469522, upper bound: 0.0455643
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457496, upper bound: 0.0476675
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0424694, 0.0427364
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0413675, 0.0417131
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0474328, upper bound: 0.0455523
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457984, upper bound: 0.0472892
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426221, 0.0426140
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415648, 0.0415553
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0478775, upper bound: 0.0452377
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460625, upper bound: 0.0463626
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0424694, 0.0427364
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0413675, 0.0417131
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0483183, upper bound: 0.0451595
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0460774, upper bound: 0.0460022
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0428051, 0.0424312
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0418023, 0.0413180
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457962, upper bound: 0.0458986
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0450206, upper bound: 0.0482309
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426666, 0.0425768
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416236, 0.0415059
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0461142, upper bound: 0.0458892
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0450880, upper bound: 0.0478105
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0428051, 0.0424312
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0418023, 0.0413180
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0473664, upper bound: 0.0457939
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457455, upper bound: 0.0475848
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426666, 0.0425768
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0416236, 0.0415059
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0477600, upper bound: 0.0457660
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0457574, upper bound: 0.0470974
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0131690, 0.0049822, -0.0131690, 0.0049822, -0.0181512, 0.0181512
1: -0.0152608, 0.0047936, -0.0152608, 0.0047936, -0.0200543, 0.0200543
2: 0.0173788, 0.0820333, 0.0173788, 0.0820333, -0.0646545, 0.0646545
3: -0.0069139, 0.0463686, -0.0069139, 0.0463686, -0.0426334, 0.0426100
4: -0.0103425, 0.0076944, -0.0103425, 0.0076944, -0.0180369, 0.0180369
5: 0.0040787, 0.0211019, 0.0040787, 0.0211019, -0.0170232, 0.0170232
6: -0.0323898, 0.0094621, -0.0323898, 0.0094621, -0.0418519, 0.0418519
7: 0.8945419, 1.0008574, 0.8945419, 1.0008574, -0.1063155, 0.1063155
8: -0.0224103, 0.0286254, -0.0224103, 0.0286254, -0.0415795, 0.0415501
9: -0.0287039, 0.0251693, -0.0287039, 0.0251693, -0.0538732, 0.0538732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 122

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0469074, upper bound: 0.0458230
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0455780, upper bound: 0.0477613
time: 0.81 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0471320, upper bound: 0.0455026
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0459555, upper bound: 0.0476713
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0476532, upper bound: 0.0454909
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460226, upper bound: 0.0472948
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0478869, upper bound: 0.0449350
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0461406, upper bound: 0.0461039
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0483557, upper bound: 0.0448971
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0461652, upper bound: 0.0457847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460202, upper bound: 0.0458064
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0453032, upper bound: 0.0481915
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0463928, upper bound: 0.0457991
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0454346, upper bound: 0.0478049
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0473615, upper bound: 0.0456007
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457932, upper bound: 0.0473339
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0477613, upper bound: 0.0455780
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0458230, upper bound: 0.0469074
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0470974, upper bound: 0.0457574
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457660, upper bound: 0.0477600
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0475848, upper bound: 0.0457455
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457939, upper bound: 0.0473664
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0478105, upper bound: 0.0450880
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0458892, upper bound: 0.0461142
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0482309, upper bound: 0.0450206
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0458986, upper bound: 0.0457962
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460022, upper bound: 0.0460774
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0451595, upper bound: 0.0483183
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0463626, upper bound: 0.0460625
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0452377, upper bound: 0.0478775
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0472892, upper bound: 0.0457984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0455523, upper bound: 0.0474328
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0476675, upper bound: 0.0457496
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0455643, upper bound: 0.0469522
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0469522, upper bound: 0.0455646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457496, upper bound: 0.0476777
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0474328, upper bound: 0.0455523
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457984, upper bound: 0.0472949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0478774, upper bound: 0.0452380
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460625, upper bound: 0.0463626
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0483183, upper bound: 0.0451597
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460774, upper bound: 0.0460022
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457962, upper bound: 0.0458986
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0450206, upper bound: 0.0482315
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0461142, upper bound: 0.0458892
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0450880, upper bound: 0.0478105
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0473611, upper bound: 0.0457939
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457432, upper bound: 0.0475848
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0477576, upper bound: 0.0457660
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457570, upper bound: 0.0470974
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0469074, upper bound: 0.0458230
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0455780, upper bound: 0.0477653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0473339, upper bound: 0.0457934
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0456007, upper bound: 0.0473664
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0478049, upper bound: 0.0454346
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457991, upper bound: 0.0463928
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0481915, upper bound: 0.0453032
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0458064, upper bound: 0.0460202
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457847, upper bound: 0.0461652
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0448971, upper bound: 0.0483558
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0461039, upper bound: 0.0461406
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0449350, upper bound: 0.0478869
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0472892, upper bound: 0.0460226
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0454887, upper bound: 0.0476532
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0476608, upper bound: 0.0459555
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0455019, upper bound: 0.0471320
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0471320, upper bound: 0.0455019
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0459555, upper bound: 0.0476608
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0476532, upper bound: 0.0454887
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460226, upper bound: 0.0472892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0478869, upper bound: 0.0449350
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0461406, upper bound: 0.0461039
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0483558, upper bound: 0.0448971
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0461652, upper bound: 0.0457847
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460202, upper bound: 0.0458064
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0453032, upper bound: 0.0481915
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0463928, upper bound: 0.0457991
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0454346, upper bound: 0.0478049
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0473664, upper bound: 0.0456007
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457934, upper bound: 0.0473339
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0477653, upper bound: 0.0455780
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0458230, upper bound: 0.0469074
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0470974, upper bound: 0.0457570
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457660, upper bound: 0.0477576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0475848, upper bound: 0.0457432
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457939, upper bound: 0.0473611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0478105, upper bound: 0.0450880
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0458892, upper bound: 0.0461142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0482315, upper bound: 0.0450206
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0458986, upper bound: 0.0457962
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460022, upper bound: 0.0460774
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0451597, upper bound: 0.0483183
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0463626, upper bound: 0.0460625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0452380, upper bound: 0.0478774
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0472949, upper bound: 0.0457984
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0455523, upper bound: 0.0474328
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0476777, upper bound: 0.0457496
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0455646, upper bound: 0.0469522
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0469522, upper bound: 0.0455643
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457496, upper bound: 0.0476675
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0474328, upper bound: 0.0455523
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457984, upper bound: 0.0472892
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0478775, upper bound: 0.0452377
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460625, upper bound: 0.0463626
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0483183, upper bound: 0.0451595
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0460774, upper bound: 0.0460022
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457962, upper bound: 0.0458986
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0450206, upper bound: 0.0482309
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0461142, upper bound: 0.0458892
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0450880, upper bound: 0.0478105
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0473664, upper bound: 0.0457939
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457455, upper bound: 0.0475848
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0477600, upper bound: 0.0457660
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0457574, upper bound: 0.0470974
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0469074, upper bound: 0.0458230
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.05
Output dim: 7, lower bound: -0.0455780, upper bound: 0.0477613
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 7, lower bound: -0.0513694, upper bound: 0.0513129
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 7, lower bound: -0.0518503, upper bound: 0.0503398
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 7, lower bound: -0.0524287, upper bound: 0.0496876
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 7, lower bound: -0.0493221, upper bound: 0.0525378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 7, lower bound: -0.0498180, upper bound: 0.0518474
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 7, lower bound: -0.0513429, upper bound: 0.0517278
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 7, lower bound: -0.0518759, upper bound: 0.0508776

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.19 + 598.13 = 601.32 seconds
