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
Threshold: 0.00217782


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0025930, -0.0011232, -0.0025930, -0.0011232, -0.0009591, 0.0009591)
1: (-0.0031215, 0.0008686, -0.0031215, 0.0008686, -0.0025175, 0.0025175)
2: (0.0040536, 0.0074761, 0.0040536, 0.0074761, -0.0021327, 0.0021327)
3: (-0.0042605, -0.0038256, -0.0042605, -0.0038256, -0.0002878, 0.0002878)
4: (0.0035345, 0.0063488, 0.0035345, 0.0063488, -0.0016257, 0.0016257)
5: (-0.0015706, 0.0013028, -0.0015706, 0.0013028, -0.0017257, 0.0017257)
6: (-0.0058728, -0.0043122, -0.0058728, -0.0043122, -0.0008510, 0.0008510)
7: (-0.0002555, 0.0025690, -0.0002555, 0.0025690, -0.0016170, 0.0016170)
8: (-0.0005452, -0.0001517, -0.0005452, -0.0001517, -0.0002688, 0.0002688)
9: (1.0022908, 1.0093822, 1.0022908, 1.0093822, -0.0044001, 0.0044001)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.35 = 2.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0025277, upper bound: 0.0025277

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0024687, upper bound: 0.0024007
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0024007, upper bound: 0.0024687
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.03 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.03
Output dim: 9, lower bound: -0.0024687, upper bound: 0.0024007
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.03
Output dim: 9, lower bound: -0.0024007, upper bound: 0.0024687

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0025930, -0.0011232, -0.0025930, -0.0011232, -0.0009230, 0.0009278
1: -0.0031215, 0.0008686, -0.0031215, 0.0008686, -0.0024460, 0.0024609
2: 0.0040536, 0.0074761, 0.0040536, 0.0074761, -0.0020933, 0.0021071
3: -0.0042605, -0.0038256, -0.0042605, -0.0038256, -0.0002687, 0.0002795
4: 0.0035345, 0.0063488, 0.0035345, 0.0063488, -0.0016139, 0.0016085
5: -0.0015706, 0.0013028, -0.0015706, 0.0013028, -0.0016821, 0.0016950
6: -0.0058728, -0.0043122, -0.0058728, -0.0043122, -0.0008503, 0.0008508
7: -0.0002555, 0.0025690, -0.0002555, 0.0025690, -0.0016170, 0.0016071
8: -0.0005452, -0.0001517, -0.0005452, -0.0001517, -0.0002578, 0.0002507
9: 1.0022908, 1.0093822, 1.0022908, 1.0093822, -0.0043319, 0.0042950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0023381, upper bound: 0.0021040
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0022101, upper bound: 0.0022906
time: 0.52 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0025930, -0.0011232, -0.0025930, -0.0011232, -0.0009278, 0.0009230
1: -0.0031215, 0.0008686, -0.0031215, 0.0008686, -0.0024609, 0.0024460
2: 0.0040536, 0.0074761, 0.0040536, 0.0074761, -0.0021071, 0.0020933
3: -0.0042605, -0.0038256, -0.0042605, -0.0038256, -0.0002795, 0.0002687
4: 0.0035345, 0.0063488, 0.0035345, 0.0063488, -0.0016085, 0.0016139
5: -0.0015706, 0.0013028, -0.0015706, 0.0013028, -0.0016950, 0.0016821
6: -0.0058728, -0.0043122, -0.0058728, -0.0043122, -0.0008508, 0.0008503
7: -0.0002555, 0.0025690, -0.0002555, 0.0025690, -0.0016071, 0.0016170
8: -0.0005452, -0.0001517, -0.0005452, -0.0001517, -0.0002507, 0.0002578
9: 1.0022908, 1.0093822, 1.0022908, 1.0093822, -0.0042950, 0.0043319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015828, upper bound: 0.0015828
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015828, upper bound: 0.0015828
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0023381, upper bound: 0.0021040
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0022101, upper bound: 0.0022906
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0015828, upper bound: 0.0015828
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0015828, upper bound: 0.0015828

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0025930, -0.0011232, -0.0025930, -0.0011232, -0.0008707, 0.0008653
1: -0.0031215, 0.0008686, -0.0031215, 0.0008686, -0.0021739, 0.0022241
2: 0.0040536, 0.0074761, 0.0040536, 0.0074761, -0.0018101, 0.0018585
3: -0.0042605, -0.0038256, -0.0042605, -0.0038256, -0.0002550, 0.0002656
4: 0.0035345, 0.0063488, 0.0035345, 0.0063488, -0.0013490, 0.0013004
5: -0.0015706, 0.0013028, -0.0015706, 0.0013028, -0.0013384, 0.0013969
6: -0.0058728, -0.0043122, -0.0058728, -0.0043122, -0.0006950, 0.0007159
7: -0.0002555, 0.0025690, -0.0002555, 0.0025690, -0.0014344, 0.0014030
8: -0.0005452, -0.0001517, -0.0005452, -0.0001517, -0.0002140, 0.0002100
9: 1.0022908, 1.0093822, 1.0022908, 1.0093822, -0.0037036, 0.0035881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0021645, upper bound: 0.0019946
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0022231, upper bound: 0.0019342
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0025930, -0.0011232, -0.0025930, -0.0011232, -0.0008606, 0.0008762
1: -0.0031215, 0.0008686, -0.0031215, 0.0008686, -0.0022112, 0.0021888
2: 0.0040536, 0.0074761, 0.0040536, 0.0074761, -0.0018458, 0.0018239
3: -0.0042605, -0.0038256, -0.0042605, -0.0038256, -0.0002548, 0.0002670
4: 0.0035345, 0.0063488, 0.0035345, 0.0063488, -0.0013058, 0.0013448
5: -0.0015706, 0.0013028, -0.0015706, 0.0013028, -0.0013877, 0.0013512
6: -0.0058728, -0.0043122, -0.0058728, -0.0043122, -0.0007155, 0.0006955
7: -0.0002555, 0.0025690, -0.0002555, 0.0025690, -0.0014129, 0.0014260
8: -0.0005452, -0.0001517, -0.0005452, -0.0001517, -0.0002173, 0.0002069
9: 1.0022908, 1.0093822, 1.0022908, 1.0093822, -0.0036250, 0.0036763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0021123, upper bound: 0.0021970
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0020751, upper bound: 0.0022137
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 9, lower bound: -0.0021645, upper bound: 0.0019946
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 9, lower bound: -0.0022231, upper bound: 0.0019342
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 9, lower bound: -0.0021123, upper bound: 0.0021970
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 9, lower bound: -0.0020751, upper bound: 0.0022137

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0025930, -0.0011232, -0.0025930, -0.0011232, -0.0005550, 0.0005236
1: -0.0031215, 0.0008686, -0.0031215, 0.0008686, -0.0010536, 0.0011733
2: 0.0040536, 0.0074761, 0.0040536, 0.0074761, -0.0010636, 0.0011422
3: -0.0042605, -0.0038256, -0.0042605, -0.0038256, -0.0002161, 0.0002243
4: 0.0035345, 0.0063488, 0.0035345, 0.0063488, -0.0010715, 0.0010230
5: -0.0015706, 0.0013028, -0.0015706, 0.0013028, -0.0009205, 0.0009823
6: -0.0058728, -0.0043122, -0.0058728, -0.0043122, -0.0006225, 0.0006408
7: -0.0002555, 0.0025690, -0.0002555, 0.0025690, -0.0011865, 0.0011570
8: -0.0005452, -0.0001517, -0.0005452, -0.0001517, -0.0001707, 0.0001679
9: 1.0022908, 1.0093822, 1.0022908, 1.0093822, -0.0023566, 0.0021835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0021390, upper bound: 0.0018378
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0021112, upper bound: 0.0018578
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0025930, -0.0011232, -0.0025930, -0.0011232, -0.0008512, 0.0008676
1: -0.0031215, 0.0008686, -0.0031215, 0.0008686, -0.0022005, 0.0021841
2: 0.0040536, 0.0074761, 0.0040536, 0.0074761, -0.0018228, 0.0018202
3: -0.0042605, -0.0038256, -0.0042605, -0.0038256, -0.0002441, 0.0002644
4: 0.0035345, 0.0063488, 0.0035345, 0.0063488, -0.0012949, 0.0013188
5: -0.0015706, 0.0013028, -0.0015706, 0.0013028, -0.0013703, 0.0013411
6: -0.0058728, -0.0043122, -0.0058728, -0.0043122, -0.0006970, 0.0006868
7: -0.0002555, 0.0025690, -0.0002555, 0.0025690, -0.0014077, 0.0013862
8: -0.0005452, -0.0001517, -0.0005452, -0.0001517, -0.0002138, 0.0001996
9: 1.0022908, 1.0093822, 1.0022908, 1.0093822, -0.0036143, 0.0036324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0019573, upper bound: 0.0020861
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0020081, upper bound: 0.0020258
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0025930, -0.0011232, -0.0025930, -0.0011232, -0.0008606, 0.0008668
1: -0.0031215, 0.0008686, -0.0031215, 0.0008686, -0.0022112, 0.0021781
2: 0.0040536, 0.0074761, 0.0040536, 0.0074761, -0.0018458, 0.0018008
3: -0.0042605, -0.0038256, -0.0042605, -0.0038256, -0.0002548, 0.0002563
4: 0.0035345, 0.0063488, 0.0035345, 0.0063488, -0.0012798, 0.0013448
5: -0.0015706, 0.0013028, -0.0015706, 0.0013028, -0.0013877, 0.0013338
6: -0.0058728, -0.0043122, -0.0058728, -0.0043122, -0.0007155, 0.0006770
7: -0.0002555, 0.0025690, -0.0002555, 0.0025690, -0.0013731, 0.0014260
8: -0.0005452, -0.0001517, -0.0005452, -0.0001517, -0.0002100, 0.0002069
9: 1.0022908, 1.0093822, 1.0022908, 1.0093822, -0.0035810, 0.0036763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0019075, upper bound: 0.0021033
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0019654, upper bound: 0.0020345
time: 0.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 9, lower bound: -0.0021390, upper bound: 0.0018378
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 9, lower bound: -0.0021112, upper bound: 0.0018578
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 9, lower bound: -0.0019573, upper bound: 0.0020861
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 9, lower bound: -0.0020081, upper bound: 0.0020258
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 9, lower bound: -0.0019075, upper bound: 0.0021033
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 9, lower bound: -0.0019654, upper bound: 0.0020345

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.67 + 19.15 = 21.82 seconds
