## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00026408


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004966, 0.0004966)
1: (-0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001883, 0.0001883)
2: (0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006663, 0.0006663)
3: (1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005649, 0.0005649)
4: (-0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0001099, 0.0001099)
5: (0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003826, 0.0003826)
6: (-0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524)
7: (-0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006809, 0.0006809)
8: (-0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0012014, 0.0012014)
9: (-0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0006042, 0.0006042)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.34 = 2.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0003419, upper bound: 0.0003419

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002967, upper bound: 0.0002967
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002967, upper bound: 0.0002967
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 3, lower bound: -0.0002967, upper bound: 0.0002967
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 3, lower bound: -0.0002967, upper bound: 0.0002967

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004951, 0.0004909
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001874, 0.0001853
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006641, 0.0006577
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005576, 0.0005627
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0001083, 0.0001095
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003815, 0.0003782
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006803, 0.0006808
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0011824, 0.0011962
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0006009, 0.0005935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0002957
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002957, upper bound: 0.0002954
time: 0.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004966, 0.0004951
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001883, 0.0001874
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006663, 0.0006641
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005627, 0.0005649
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0001095, 0.0001099
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003826, 0.0003815
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006808, 0.0006809
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0011962, 0.0012014
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0006042, 0.0006009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0002957
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002957, upper bound: 0.0002954
time: 0.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0002957
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 3, lower bound: -0.0002957, upper bound: 0.0002954
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0002957
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 3, lower bound: -0.0002957, upper bound: 0.0002954

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004899, 0.0004856
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001838, 0.0001817
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006561, 0.0006496
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005497, 0.0005549
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0001067, 0.0001079
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003773, 0.0003740
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006797, 0.0006802
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0011602, 0.0011743
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005882, 0.0005806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0002909
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0002909
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004898, 0.0004857
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001837, 0.0001817
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006560, 0.0006497
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005498, 0.0005548
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0001067, 0.0001079
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003773, 0.0003741
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006797, 0.0006802
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0011604, 0.0011740
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005880, 0.0005807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002909, upper bound: 0.0002907
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002909, upper bound: 0.0002906
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004914, 0.0004898
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001846, 0.0001837
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006585, 0.0006560
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005548, 0.0005570
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0001079, 0.0001084
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003785, 0.0003773
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006802, 0.0006803
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0011740, 0.0011800
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005913, 0.0005880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0002909
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0002909
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004913, 0.0004899
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001846, 0.0001838
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006583, 0.0006561
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005549, 0.0005569
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0001079, 0.0001084
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003784, 0.0003773
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006802, 0.0006803
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0011743, 0.0011797
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005912, 0.0005882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002909, upper bound: 0.0002907
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002909, upper bound: 0.0002906
time: 0.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.32
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0002909
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.32
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0002909
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.32
Output dim: 3, lower bound: -0.0002909, upper bound: 0.0002907
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.32
Output dim: 3, lower bound: -0.0002909, upper bound: 0.0002906
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.32
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0002909
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.32
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0002909
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.32
Output dim: 3, lower bound: -0.0002909, upper bound: 0.0002907
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.32
Output dim: 3, lower bound: -0.0002909, upper bound: 0.0002906

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004723, 0.0004640
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001716, 0.0001675
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006235, 0.0006107
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005194, 0.0005296
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000983, 0.0001007
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003631, 0.0003565
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006780, 0.0006790
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010584, 0.0010862
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005402, 0.0005253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002660
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002660
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004682, 0.0004681
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001696, 0.0001696
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006172, 0.0006171
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005245, 0.0005246
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000995, 0.0000996
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003598, 0.0003598
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006785, 0.0006785
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010722, 0.0010725
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005329, 0.0005327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004717, 0.0004641
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001713, 0.0001676
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006226, 0.0006108
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005195, 0.0005288
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000984, 0.0001006
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003626, 0.0003566
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006780, 0.0006789
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010586, 0.0010841
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005391, 0.0005255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004681, 0.0004688
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001696, 0.0001699
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006171, 0.0006181
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005253, 0.0005245
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000997, 0.0000995
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003598, 0.0003603
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006785, 0.0006785
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010745, 0.0010722
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005327, 0.0005339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002660, upper bound: 0.0002659
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002660, upper bound: 0.0002659
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004735, 0.0004681
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001728, 0.0001696
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006257, 0.0006171
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005245, 0.0005324
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000995, 0.0001012
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003640, 0.0003598
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006785, 0.0006791
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010722, 0.0010924
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005442, 0.0005327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002660
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002660
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004694, 0.0004717
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001707, 0.0001713
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006194, 0.0006226
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005288, 0.0005274
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0001006, 0.0001000
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003608, 0.0003626
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006789, 0.0006786
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010841, 0.0010787
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005369, 0.0005391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004729, 0.0004682
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001725, 0.0001696
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006248, 0.0006172
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005246, 0.0005317
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000996, 0.0001010
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003635, 0.0003598
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006785, 0.0006790
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010725, 0.0010903
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005431, 0.0005329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004693, 0.0004723
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001707, 0.0001716
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006193, 0.0006235
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005296, 0.0005273
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0001007, 0.0001000
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003607, 0.0003631
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006790, 0.0006786
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010862, 0.0010784
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005368, 0.0005402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002660, upper bound: 0.0002659
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002660, upper bound: 0.0002659
time: 0.52 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002660
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002660
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002660, upper bound: 0.0002659
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002660, upper bound: 0.0002659
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002660
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002660
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002659, upper bound: 0.0002659
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002660, upper bound: 0.0002659
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -0.0002660, upper bound: 0.0002659

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004695, 0.0004609
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001700, 0.0001658
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006189, 0.0006057
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005152, 0.0005256
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000973, 0.0000998
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003608, 0.0003541
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006776, 0.0006786
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010462, 0.0010749
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005344, 0.0005191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002560, upper bound: 0.0002562
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002558
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004723, 0.0004611
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001716, 0.0001659
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006235, 0.0006061
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005155, 0.0005296
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000974, 0.0001007
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003631, 0.0003543
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006776, 0.0006790
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010470, 0.0010862
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005402, 0.0005195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002560, upper bound: 0.0002562
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002558
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004654, 0.0004649
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001680, 0.0001678
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006126, 0.0006118
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005200, 0.0005206
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000984, 0.0000986
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003576, 0.0003572
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006781, 0.0006781
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010595, 0.0010611
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005271, 0.0005262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002559, upper bound: 0.0002562
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002563, upper bound: 0.0002558
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004682, 0.0004653
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001696, 0.0001680
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006172, 0.0006124
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005205, 0.0005246
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000986, 0.0000996
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003598, 0.0003575
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006781, 0.0006785
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010608, 0.0010725
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005329, 0.0005269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002559, upper bound: 0.0002562
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002563, upper bound: 0.0002558
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004689, 0.0004610
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001697, 0.0001659
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006180, 0.0006058
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005152, 0.0005249
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000973, 0.0000996
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003604, 0.0003541
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006776, 0.0006786
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010465, 0.0010728
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005333, 0.0005192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002563
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002559
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004717, 0.0004612
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001713, 0.0001660
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006226, 0.0006062
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005155, 0.0005288
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000974, 0.0001006
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003626, 0.0003543
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006777, 0.0006789
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010473, 0.0010841
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005391, 0.0005197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002563
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002559
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004653, 0.0004655
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001680, 0.0001681
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006125, 0.0006128
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005208, 0.0005205
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000986, 0.0000986
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003575, 0.0003577
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006782, 0.0006781
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010617, 0.0010609
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005269, 0.0005274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002562
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002560
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004681, 0.0004660
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001696, 0.0001683
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006171, 0.0006135
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005213, 0.0005245
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000987, 0.0000995
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003598, 0.0003581
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006782, 0.0006785
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010631, 0.0010722
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005327, 0.0005281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002562
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002560
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004706, 0.0004647
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001712, 0.0001677
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006209, 0.0006116
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005198, 0.0005285
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000984, 0.0001002
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003618, 0.0003571
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006781, 0.0006788
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010590, 0.0010811
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005385, 0.0005260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002560, upper bound: 0.0002562
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002558
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004735, 0.0004653
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001728, 0.0001680
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006257, 0.0006125
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005205, 0.0005324
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000986, 0.0001012
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003640, 0.0003575
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006781, 0.0006791
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010609, 0.0010924
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005442, 0.0005269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002560, upper bound: 0.0002562
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002558
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004665, 0.0004684
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001692, 0.0001695
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006146, 0.0006172
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005243, 0.0005235
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000994, 0.0000990
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003585, 0.0003600
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006785, 0.0006783
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010711, 0.0010673
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005311, 0.0005324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002559, upper bound: 0.0002562
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002563, upper bound: 0.0002558
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004694, 0.0004689
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001707, 0.0001697
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006194, 0.0006180
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005249, 0.0005274
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000996, 0.0001000
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003608, 0.0003604
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006786, 0.0006786
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010728, 0.0010787
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005369, 0.0005333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002559, upper bound: 0.0002562
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002563, upper bound: 0.0002558
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004700, 0.0004650
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001709, 0.0001678
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006200, 0.0006120
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005201, 0.0005278
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000985, 0.0001000
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003613, 0.0003573
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006781, 0.0006787
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010598, 0.0010790
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005373, 0.0005264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002563
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002559
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004729, 0.0004654
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001725, 0.0001680
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006248, 0.0006126
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005206, 0.0005317
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000986, 0.0001010
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003635, 0.0003576
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006781, 0.0006790
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010611, 0.0010903
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005431, 0.0005271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002563
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002559
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004664, 0.0004691
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001691, 0.0001698
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006145, 0.0006182
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005251, 0.0005234
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000996, 0.0000990
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003584, 0.0003605
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006786, 0.0006783
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010734, 0.0010671
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005310, 0.0005336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002562
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002560
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0010800, -0.0004502, -0.0010800, -0.0004502, -0.0004693, 0.0004695
1: -0.0042227, -0.0039952, -0.0042227, -0.0039952, -0.0001707, 0.0001700
2: 0.0131166, 0.0139855, 0.0131166, 0.0139855, -0.0006193, 0.0006189
3: 1.0084403, 1.0090206, 1.0084403, 1.0090206, -0.0005256, 0.0005273
4: -0.0038714, -0.0037245, -0.0038714, -0.0037245, -0.0000998, 0.0001000
5: 0.0031164, 0.0036037, 0.0031164, 0.0036037, -0.0003607, 0.0003608
6: -0.0024350, -0.0023826, -0.0024350, -0.0023826, -0.0000524, 0.0000524
7: -0.0129430, -0.0122493, -0.0129430, -0.0122493, -0.0006786, 0.0006786
8: -0.0092745, -0.0076517, -0.0092745, -0.0076517, -0.0010749, 0.0010784
9: -0.0005695, 0.0002524, -0.0005695, 0.0002524, -0.0005368, 0.0005344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002562
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002560
time: 0.48 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002560, upper bound: 0.0002562
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002558
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002560, upper bound: 0.0002562
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002558
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002559, upper bound: 0.0002562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002563, upper bound: 0.0002558
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002559, upper bound: 0.0002562
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002563, upper bound: 0.0002558
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002563
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002563
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002559
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002562
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002560
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002562
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002560
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002560, upper bound: 0.0002562
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002558
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002560, upper bound: 0.0002562
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002558
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002559, upper bound: 0.0002562
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002563, upper bound: 0.0002558
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002559, upper bound: 0.0002562
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002563, upper bound: 0.0002558
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002563
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002559
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002563
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002559
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002562
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002560
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002558, upper bound: 0.0002562
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.60
Output dim: 3, lower bound: -0.0002562, upper bound: 0.0002560

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.88 + 137.34 = 140.23 seconds
