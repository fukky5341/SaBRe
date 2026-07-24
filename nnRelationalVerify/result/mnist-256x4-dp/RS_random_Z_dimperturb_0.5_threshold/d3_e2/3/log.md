## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.37300728


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206)
1: (-0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397)
2: (-0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760)
3: (-0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4852566, 0.4852564)
4: (-0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662)
5: (-0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523)
6: (-0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901)
7: (0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507)
8: (-0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192)
9: (-0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 2.08 = 3.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.5201641, upper bound: 0.5201641

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5198676, upper bound: 0.5196774
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5196774, upper bound: 0.5198676
time: 1.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.04
Output dim: 7, lower bound: -0.5198676, upper bound: 0.5196774
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.04
Output dim: 7, lower bound: -0.5196774, upper bound: 0.5198676

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4851099, 0.4851570
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4717385, upper bound: 0.4716752
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4717385, upper bound: 0.4716752
time: 0.97 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4851571, 0.4851097
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4765232, upper bound: 0.4767229
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4765232, upper bound: 0.4767229
time: 0.86 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 7, lower bound: -0.4717385, upper bound: 0.4716752
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 7, lower bound: -0.4717385, upper bound: 0.4716752
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 7, lower bound: -0.4765232, upper bound: 0.4767229
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 7, lower bound: -0.4765232, upper bound: 0.4767229

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4848580, 0.4849535
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4650232, upper bound: 0.4649325
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4650639, upper bound: 0.4648890
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4849063, 0.4851570
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4691704, upper bound: 0.4693148
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4693816, upper bound: 0.4690355
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4851270, 0.4850937
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4761964, upper bound: 0.4740722
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4736886, upper bound: 0.4763979
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4851409, 0.4851097
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4751843, upper bound: 0.4741096
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740714, upper bound: 0.4753798
time: 0.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.4650232, upper bound: 0.4649325
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.4650639, upper bound: 0.4648890
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.4691704, upper bound: 0.4693148
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.4693816, upper bound: 0.4690355
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.4761964, upper bound: 0.4740722
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.4736886, upper bound: 0.4763979
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.4751843, upper bound: 0.4741096
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.4740714, upper bound: 0.4753798

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845381, 0.4846321
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3995817, upper bound: 0.3995879
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3995817, upper bound: 0.3995879
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845366, 0.4849535
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4627170, upper bound: 0.4627414
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4629395, upper bound: 0.4624969
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845343, 0.4847287
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4691040, upper bound: 0.4687177
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4684367, upper bound: 0.4692449
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844781, 0.4847921
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4691192, upper bound: 0.4664003
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4666826, upper bound: 0.4687708
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4851273, 0.4850989
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4760890, upper bound: 0.4736054
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4757011, upper bound: 0.4739638
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4851323, 0.4850940
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4647100, upper bound: 0.4675708
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4647100, upper bound: 0.4675708
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4848062, 0.4848416
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4721967, upper bound: 0.4710645
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4722150, upper bound: 0.4710609
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4848753, 0.4847750
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4651734, upper bound: 0.4665575
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4651734, upper bound: 0.4665575
time: 0.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.3995817, upper bound: 0.3995879
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.3995817, upper bound: 0.3995879
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4627170, upper bound: 0.4627414
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4629395, upper bound: 0.4624969
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4691040, upper bound: 0.4687177
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4684367, upper bound: 0.4692449
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4691192, upper bound: 0.4664003
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4666826, upper bound: 0.4687708
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4760890, upper bound: 0.4736054
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4757011, upper bound: 0.4739638
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4647100, upper bound: 0.4675708
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4647100, upper bound: 0.4675708
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4721967, upper bound: 0.4710645
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4722150, upper bound: 0.4710609
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4651734, upper bound: 0.4665575
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 7, lower bound: -0.4651734, upper bound: 0.4665575

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4852568, 0.4840079
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3993353, upper bound: 0.3992289
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3992286, upper bound: 0.3993412
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4839138, 0.4846321
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3987394, upper bound: 0.3966997
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3966881, upper bound: 0.3987454
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841816, 0.4845253
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4624133, upper bound: 0.4600660
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4601532, upper bound: 0.4624546
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841063, 0.4845886
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4628440, upper bound: 0.4616879
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4624570, upper bound: 0.4624051
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844863, 0.4846934
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4687991, upper bound: 0.4662211
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4663932, upper bound: 0.4684075
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844987, 0.4846745
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4681274, upper bound: 0.4665040
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4660694, upper bound: 0.4689422
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844781, 0.4847972
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4053222, upper bound: 0.4030246
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4053222, upper bound: 0.4030246
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844831, 0.4847922
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4602160, upper bound: 0.4622460
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4602461, upper bound: 0.4622048
time: 2.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4850714, 0.4850619
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4760890, upper bound: 0.4734563
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4755956, upper bound: 0.4736054
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4850904, 0.4850416
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4666414, upper bound: 0.4651043
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4666414, upper bound: 0.4651043
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847863, 0.4847729
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4647100, upper bound: 0.4669176
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4644960, upper bound: 0.4675708
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4848112, 0.4850940
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4647100, upper bound: 0.4669176
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4644960, upper bound: 0.4675708
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844289, 0.4844012
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4720910, upper bound: 0.4707557
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4716278, upper bound: 0.4709564
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4843655, 0.4844666
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4215828, upper bound: 0.4214310
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4215828, upper bound: 0.4214310
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845352, 0.4844526
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4650608, upper bound: 0.4659769
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4645323, upper bound: 0.4664443
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845530, 0.4847750
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619852, upper bound: 0.4634046
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619856, upper bound: 0.4634056
time: 1.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.3993353, upper bound: 0.3992289
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.3992286, upper bound: 0.3993412
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.3987394, upper bound: 0.3966997
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.3966881, upper bound: 0.3987454
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4624133, upper bound: 0.4600660
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4601532, upper bound: 0.4624546
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4628440, upper bound: 0.4616879
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4624570, upper bound: 0.4624051
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4687991, upper bound: 0.4662211
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4663932, upper bound: 0.4684075
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4681274, upper bound: 0.4665040
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4660694, upper bound: 0.4689422
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4053222, upper bound: 0.4030246
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4053222, upper bound: 0.4030246
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4602160, upper bound: 0.4622460
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4602461, upper bound: 0.4622048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4760890, upper bound: 0.4734563
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4755956, upper bound: 0.4736054
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4666414, upper bound: 0.4651043
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4666414, upper bound: 0.4651043
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4647100, upper bound: 0.4669176
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4644960, upper bound: 0.4675708
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4647100, upper bound: 0.4669176
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4644960, upper bound: 0.4675708
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4720910, upper bound: 0.4707557
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4716278, upper bound: 0.4709564
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4215828, upper bound: 0.4214310
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4215828, upper bound: 0.4214310
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4650608, upper bound: 0.4659769
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4645323, upper bound: 0.4664443
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4619852, upper bound: 0.4634046
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.29
Output dim: 7, lower bound: -0.4619856, upper bound: 0.4634056

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4852040, 0.4839707
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3962139, upper bound: 0.3957827
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3958798, upper bound: 0.3960972
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4852195, 0.4839518
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3983865, upper bound: 0.3964556
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3960175, upper bound: 0.3984988
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4839145, 0.4846375
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3956141, upper bound: 0.3932678
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3952837, upper bound: 0.3935829
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4839190, 0.4846326
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3941224, upper bound: 0.3962525
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3942083, upper bound: 0.3959908
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841820, 0.4845303
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4623062, upper bound: 0.4596660
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4616900, upper bound: 0.4599568
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841868, 0.4845254
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4601461, upper bound: 0.4623180
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4601532, upper bound: 0.4624546
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840577, 0.4845532
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4628440, upper bound: 0.4615941
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4627610, upper bound: 0.4616879
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840713, 0.4845409
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4621477, upper bound: 0.4598302
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4598346, upper bound: 0.4620993
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844865, 0.4846985
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4676188, upper bound: 0.4638573
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4664379, upper bound: 0.4650438
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844916, 0.4846936
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4651817, upper bound: 0.4660329
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4642470, upper bound: 0.4671860
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844990, 0.4846796
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4681274, upper bound: 0.4665040
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4680442, upper bound: 0.4665025
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845040, 0.4846746
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4660694, upper bound: 0.4688572
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4660609, upper bound: 0.4689422
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4851629, 0.4841729
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4053222, upper bound: 0.4030167
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050852, upper bound: 0.4030246
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4838539, 0.4847972
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4053222, upper bound: 0.4030167
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050852, upper bound: 0.4030246
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841269, 0.4844692
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4602154, upper bound: 0.4621389
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4602160, upper bound: 0.4622460
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841598, 0.4847922
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4601362, upper bound: 0.4613780
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4598346, upper bound: 0.4620993
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4850328, 0.4850818
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4747519, upper bound: 0.4706169
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4736410, upper bound: 0.4721241
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4850910, 0.4850234
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4725976, upper bound: 0.4706066
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726557, upper bound: 0.4704987
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847448, 0.4847209
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3724292, upper bound: 0.3713325
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3724292, upper bound: 0.3713325
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847696, 0.4850416
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4653211, upper bound: 0.4625002
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4642071, upper bound: 0.4637843
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847474, 0.4847924
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4615849, upper bound: 0.4638449
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4615953, upper bound: 0.4638142
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4848021, 0.4847340
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3712658, upper bound: 0.3726225
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3712658, upper bound: 0.3726225
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847723, 0.4851142
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4633893, upper bound: 0.4647163
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4624247, upper bound: 0.4655679
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4848305, 0.4850558
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4643852, upper bound: 0.4669843
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4639377, upper bound: 0.4674594
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4843806, 0.4843653
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4629471, upper bound: 0.4615156
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4629471, upper bound: 0.4615156
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4843929, 0.4843451
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4624761, upper bound: 0.4619280
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4624761, upper bound: 0.4619280
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4850557, 0.4838422
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4210796, upper bound: 0.4182357
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4186088, upper bound: 0.4208880
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4837414, 0.4844666
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058545, upper bound: 0.4054882
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058545, upper bound: 0.4054882
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844840, 0.4844182
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4650608, upper bound: 0.4653647
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4648046, upper bound: 0.4659769
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845008, 0.4843980
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4642071, upper bound: 0.4637843
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4616444, upper bound: 0.4661225
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841762, 0.4843345
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619852, upper bound: 0.4628211
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4617823, upper bound: 0.4634046
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841104, 0.4843906
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619856, upper bound: 0.4628083
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4618045, upper bound: 0.4634056
time: 0.99 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3962139, upper bound: 0.3957827
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3958798, upper bound: 0.3960972
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3983865, upper bound: 0.3964556
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3960175, upper bound: 0.3984988
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3956141, upper bound: 0.3932678
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3952837, upper bound: 0.3935829
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3941224, upper bound: 0.3962525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3942083, upper bound: 0.3959908
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4623062, upper bound: 0.4596660
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4616900, upper bound: 0.4599568
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4601461, upper bound: 0.4623180
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4601532, upper bound: 0.4624546
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4628440, upper bound: 0.4615941
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4627610, upper bound: 0.4616879
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4621477, upper bound: 0.4598302
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4598346, upper bound: 0.4620993
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4676188, upper bound: 0.4638573
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4664379, upper bound: 0.4650438
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4651817, upper bound: 0.4660329
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4642470, upper bound: 0.4671860
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4681274, upper bound: 0.4665040
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4680442, upper bound: 0.4665025
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4660694, upper bound: 0.4688572
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4660609, upper bound: 0.4689422
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4053222, upper bound: 0.4030167
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4050852, upper bound: 0.4030246
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4053222, upper bound: 0.4030167
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4050852, upper bound: 0.4030246
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4602154, upper bound: 0.4621389
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4602160, upper bound: 0.4622460
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4601362, upper bound: 0.4613780
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4598346, upper bound: 0.4620993
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4747519, upper bound: 0.4706169
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4736410, upper bound: 0.4721241
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4725976, upper bound: 0.4706066
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4726557, upper bound: 0.4704987
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3724292, upper bound: 0.3713325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3724292, upper bound: 0.3713325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4653211, upper bound: 0.4625002
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4642071, upper bound: 0.4637843
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4615849, upper bound: 0.4638449
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4615953, upper bound: 0.4638142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3712658, upper bound: 0.3726225
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.3712658, upper bound: 0.3726225
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4633893, upper bound: 0.4647163
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4624247, upper bound: 0.4655679
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4643852, upper bound: 0.4669843
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4639377, upper bound: 0.4674594
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4629471, upper bound: 0.4615156
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4629471, upper bound: 0.4615156
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4624761, upper bound: 0.4619280
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4624761, upper bound: 0.4619280
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4210796, upper bound: 0.4182357
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4186088, upper bound: 0.4208880
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4058545, upper bound: 0.4054882
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4058545, upper bound: 0.4054882
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4650608, upper bound: 0.4653647
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4648046, upper bound: 0.4659769
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4642071, upper bound: 0.4637843
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4616444, upper bound: 0.4661225
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4619852, upper bound: 0.4628211
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4617823, upper bound: 0.4634046
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4619856, upper bound: 0.4628083
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 7, lower bound: -0.4618045, upper bound: 0.4634056

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4848711, 0.4837071
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3962139, upper bound: 0.3952092
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3957186, upper bound: 0.3957827
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4849688, 0.4836378
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3950372, upper bound: 0.3929185
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3930234, upper bound: 0.3952549
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4852202, 0.4839569
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3954208, upper bound: 0.3939728
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3958817, upper bound: 0.3938855
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4852246, 0.4839519
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3929190, upper bound: 0.3950429
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3924431, upper bound: 0.3953731
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4835792, 0.4843705
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3956141, upper bound: 0.3931983
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3951123, upper bound: 0.3932678
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4836388, 0.4843012
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3926764, upper bound: 0.3910995
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3927949, upper bound: 0.3909936
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4835575, 0.4842021
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3938800, upper bound: 0.3958789
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3933697, upper bound: 0.3960060
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4834878, 0.4842653
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3939644, upper bound: 0.3954109
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3935453, upper bound: 0.3957470
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841356, 0.4844950
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4623062, upper bound: 0.4596648
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4622603, upper bound: 0.4596660
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841470, 0.4844761
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3954208, upper bound: 0.3939728
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3954208, upper bound: 0.3939728
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841441, 0.4845423
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3941224, upper bound: 0.3958029
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3941224, upper bound: 0.3958029
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4842017, 0.4844835
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4589116, upper bound: 0.4600942
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4579269, upper bound: 0.4612232
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840147, 0.4845662
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3968481, upper bound: 0.3957386
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3968481, upper bound: 0.3957386
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840753, 0.4845110
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4624551, upper bound: 0.4593244
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4601362, upper bound: 0.4613780
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840714, 0.4845461
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3958817, upper bound: 0.3938855
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3958817, upper bound: 0.3938855
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840764, 0.4845410
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3935453, upper bound: 0.3957470
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3935453, upper bound: 0.3957470
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841390, 0.4844203
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4018104, upper bound: 0.3988058
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4018104, upper bound: 0.3988058
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4842153, 0.4843510
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4015024, upper bound: 0.3994839
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4015024, upper bound: 0.3994839
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841441, 0.4844153
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4587963, upper bound: 0.4595784
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4587984, upper bound: 0.4594942
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4842205, 0.4843462
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4577340, upper bound: 0.4606865
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4578053, upper bound: 0.4606780
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844565, 0.4846963
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4616844, upper bound: 0.4599563
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4616900, upper bound: 0.4599541
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845135, 0.4846373
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4668621, upper bound: 0.4642870
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4656881, upper bound: 0.4653057
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844615, 0.4846910
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4595975, upper bound: 0.4623070
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4595987, upper bound: 0.4622089
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845185, 0.4846323
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4595952, upper bound: 0.4623995
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4595976, upper bound: 0.4623448
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4851210, 0.4841861
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050774, upper bound: 0.4020012
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4049115, upper bound: 0.4027730
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4851773, 0.4841309
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4048410, upper bound: 0.4022605
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4044456, upper bound: 0.4027801
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4838121, 0.4848105
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4022040, upper bound: 0.3997263
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4017866, upper bound: 0.3998674
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4838705, 0.4847553
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4019623, upper bound: 0.3997376
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4016737, upper bound: 0.3998754
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840841, 0.4844816
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3942083, upper bound: 0.3956586
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3942083, upper bound: 0.3956586
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841392, 0.4844265
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4589938, upper bound: 0.4600476
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4578961, upper bound: 0.4610296
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841045, 0.4847569
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4589057, upper bound: 0.4591920
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4578757, upper bound: 0.4601677
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4841248, 0.4847445
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4598340, upper bound: 0.4619726
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4598346, upper bound: 0.4620993
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847016, 0.4848136
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4717695, upper bound: 0.4675934
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4717875, upper bound: 0.4674916
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847701, 0.4847506
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4240926, upper bound: 0.4205121
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4240926, upper bound: 0.4205121
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847264, 0.4845927
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4226032, upper bound: 0.4203390
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4226032, upper bound: 0.4203390
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4846603, 0.4846488
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4712792, upper bound: 0.4675185
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4703932, upper bound: 0.4691756
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844358, 0.4847763
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4653211, upper bound: 0.4624835
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4647842, upper bound: 0.4625002
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845053, 0.4847096
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3709530, upper bound: 0.3699968
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3709530, upper bound: 0.3699968
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4843831, 0.4843581
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3682526, upper bound: 0.3695366
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3682526, upper bound: 0.3695366
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4843131, 0.4844127
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057117, upper bound: 0.4067341
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057117, upper bound: 0.4067341
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4844377, 0.4848439
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4632745, upper bound: 0.4640556
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4627370, upper bound: 0.4646032
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4845071, 0.4847810
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3699111, upper bound: 0.3710345
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3699111, upper bound: 0.3710345
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847745, 0.4850184
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4630228, upper bound: 0.4643534
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4622531, upper bound: 0.4656578
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4847933, 0.4849982
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4607871, upper bound: 0.4642962
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4608592, upper bound: 0.4642962
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840364, 0.4840410
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3682823, upper bound: 0.3684864
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3682823, upper bound: 0.3684864
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840562, 0.4843653
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057506, upper bound: 0.4046680
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057506, upper bound: 0.4046680
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840509, 0.4840209
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4624761, upper bound: 0.4617573
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619904, upper bound: 0.4619280
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2462979, 0.1936227, -0.2462979, 0.1936227, -0.4399206, 0.4399206
1: -0.2266728, 0.1924669, -0.2266728, 0.1924669, -0.4191397, 0.4191397
2: -0.1665448, 0.2945312, -0.1665448, 0.2945312, -0.4610760, 0.4610760
3: -0.1245037, 0.3646154, -0.1245037, 0.3646154, -0.4840686, 0.4843451
4: -0.2002433, 0.2527229, -0.2002433, 0.2527229, -0.4529662, 0.4529662
5: -0.1984656, 0.2807867, -0.1984656, 0.2807867, -0.4792523, 0.4792523
6: -0.2389758, 0.2352142, -0.2389758, 0.2352142, -0.4741901, 0.4741901
7: 0.4576564, 1.1032071, 0.4576564, 1.1032071, -0.6455507, 0.6455507
8: -0.1891579, 0.3165613, -0.1891579, 0.3165613, -0.5057192, 0.5057192
9: -0.1869313, 0.3043408, -0.1869313, 0.3043408, -0.4912720, 0.4912720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.58 + 598.04 = 601.62 seconds
