## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00285264


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798)
1: (0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632)
2: (-0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342)
3: (0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275)
4: (0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269)
5: (0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133)
6: (-0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189)
7: (-0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150)
8: (0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048832, 0.0048832)
9: (-0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.17 + 2.92 = 4.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0032413, upper bound: 0.0032413

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032180, upper bound: 0.0032205
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032180, upper bound: 0.0032180
time: 1.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.68
Output dim: 1, lower bound: -0.0032180, upper bound: 0.0032205
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.68
Output dim: 1, lower bound: -0.0032180, upper bound: 0.0032180

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048751, 0.0048743
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031597, upper bound: 0.0031786
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031735, upper bound: 0.0031600
time: 1.22 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048743, 0.0048751
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032186, upper bound: 0.0032161
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032161, upper bound: 0.0032162
time: 1.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.58 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 1, lower bound: -0.0031597, upper bound: 0.0031786
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 1, lower bound: -0.0031735, upper bound: 0.0031600
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 1, lower bound: -0.0032186, upper bound: 0.0032161
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 1, lower bound: -0.0032161, upper bound: 0.0032162

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048652, 0.0048692
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031431, upper bound: 0.0031656
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031431, upper bound: 0.0031656
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048751, 0.0048644
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031651, upper bound: 0.0031488
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031483, upper bound: 0.0031514
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048715, 0.0048717
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032046, upper bound: 0.0032048
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032046, upper bound: 0.0032077
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048705, 0.0048723
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031577, upper bound: 0.0031715
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031760, upper bound: 0.0031576
time: 1.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.79 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 1, lower bound: -0.0031431, upper bound: 0.0031656
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 1, lower bound: -0.0031431, upper bound: 0.0031656
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 1, lower bound: -0.0031651, upper bound: 0.0031488
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 1, lower bound: -0.0031483, upper bound: 0.0031514
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 1, lower bound: -0.0032046, upper bound: 0.0032048
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 1, lower bound: -0.0032046, upper bound: 0.0032077
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 1, lower bound: -0.0031577, upper bound: 0.0031715
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 1, lower bound: -0.0031760, upper bound: 0.0031576

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048628, 0.0048673
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030292, upper bound: 0.0030589
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030292, upper bound: 0.0031112
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048624, 0.0048669
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031410, upper bound: 0.0031627
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031410, upper bound: 0.0031635
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048762, 0.0048638
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031536, upper bound: 0.0031414
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031401, upper bound: 0.0031444
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048745, 0.0048654
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031169, upper bound: 0.0030954
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031022, upper bound: 0.0031141
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048723, 0.0048710
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031459, upper bound: 0.0031598
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031597, upper bound: 0.0031463
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048708, 0.0048722
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032007, upper bound: 0.0032055
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032051, upper bound: 0.0032042
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048606, 0.0048648
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031459, upper bound: 0.0031597
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031459, upper bound: 0.0031631
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048705, 0.0048624
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031733, upper bound: 0.0031556
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031741, upper bound: 0.0031551
time: 1.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0030292, upper bound: 0.0030589
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0030292, upper bound: 0.0031112
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031410, upper bound: 0.0031627
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031410, upper bound: 0.0031635
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031536, upper bound: 0.0031414
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031401, upper bound: 0.0031444
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031169, upper bound: 0.0030954
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031022, upper bound: 0.0031141
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031459, upper bound: 0.0031598
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031597, upper bound: 0.0031463
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0032007, upper bound: 0.0032055
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0032051, upper bound: 0.0032042
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031459, upper bound: 0.0031597
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031459, upper bound: 0.0031631
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031733, upper bound: 0.0031556
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0031741, upper bound: 0.0031551

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048497, 0.0048458
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027762, upper bound: 0.0027510
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027762, upper bound: 0.0027510
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048413, 0.0048532
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027486, upper bound: 0.0027778
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027486, upper bound: 0.0027778
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048593, 0.0048618
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030832, upper bound: 0.0030954
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030832, upper bound: 0.0031402
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048588, 0.0048638
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031318, upper bound: 0.0031534
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031318, upper bound: 0.0031593
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048748, 0.0048616
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027593, upper bound: 0.0027626
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027593, upper bound: 0.0027626
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048739, 0.0048624
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027574, upper bound: 0.0027633
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027574, upper bound: 0.0027633
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048752, 0.0048612
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024494, upper bound: 0.0024535
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024494, upper bound: 0.0024535
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048703, 0.0048654
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030793, upper bound: 0.0031015
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030793, upper bound: 0.0031015
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048625, 0.0048629
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031467, upper bound: 0.0031578
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031473, upper bound: 0.0031568
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048723, 0.0048611
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028300, upper bound: 0.0028262
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028300, upper bound: 0.0028262
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048684, 0.0048697
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031872, upper bound: 0.0031917
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031872, upper bound: 0.0031923
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048683, 0.0048698
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031363, upper bound: 0.0031439
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031363, upper bound: 0.0031822
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048611, 0.0048642
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030294, upper bound: 0.0030482
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030357, upper bound: 0.0031068
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048599, 0.0048658
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030294, upper bound: 0.0030489
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030294, upper bound: 0.0031107
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048691, 0.0048607
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031701, upper bound: 0.0031533
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031710, upper bound: 0.0031510
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048691, 0.0048610
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030968, upper bound: 0.0030969
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031079, upper bound: 0.0031319
time: 1.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0027762, upper bound: 0.0027510
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0027762, upper bound: 0.0027510
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0027486, upper bound: 0.0027778
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0027486, upper bound: 0.0027778
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0030832, upper bound: 0.0030954
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0030832, upper bound: 0.0031402
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031318, upper bound: 0.0031534
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031318, upper bound: 0.0031593
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0027593, upper bound: 0.0027626
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0027593, upper bound: 0.0027626
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0027574, upper bound: 0.0027633
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0027574, upper bound: 0.0027633
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0024494, upper bound: 0.0024535
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0024494, upper bound: 0.0024535
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0030793, upper bound: 0.0031015
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0030793, upper bound: 0.0031015
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031467, upper bound: 0.0031578
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031473, upper bound: 0.0031568
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0028300, upper bound: 0.0028262
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0028300, upper bound: 0.0028262
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031872, upper bound: 0.0031917
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031872, upper bound: 0.0031923
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031363, upper bound: 0.0031439
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031363, upper bound: 0.0031822
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0030294, upper bound: 0.0030482
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0030357, upper bound: 0.0031068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0030294, upper bound: 0.0030489
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0030294, upper bound: 0.0031107
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031701, upper bound: 0.0031533
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031710, upper bound: 0.0031510
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0030968, upper bound: 0.0030969
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 1, lower bound: -0.0031079, upper bound: 0.0031319

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048559, 0.0048550
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030798, upper bound: 0.0030930
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030798, upper bound: 0.0030917
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048524, 0.0048588
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030809, upper bound: 0.0031381
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030808, upper bound: 0.0031371
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048581, 0.0048624
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030952, upper bound: 0.0030981
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030897, upper bound: 0.0031021
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048575, 0.0048631
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031197, upper bound: 0.0031476
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031197, upper bound: 0.0031508
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048699, 0.0048648
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030164, upper bound: 0.0030456
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030164, upper bound: 0.0030784
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048697, 0.0048654
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030276, upper bound: 0.0030435
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030325, upper bound: 0.0030461
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048609, 0.0048612
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031313, upper bound: 0.0031441
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031253, upper bound: 0.0031440
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048607, 0.0048613
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027631, upper bound: 0.0027564
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027631, upper bound: 0.0027564
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048660, 0.0048670
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031831, upper bound: 0.0031832
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031795, upper bound: 0.0031876
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048658, 0.0048673
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031262, upper bound: 0.0031337
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031262, upper bound: 0.0031543
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048648, 0.0048630
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031208, upper bound: 0.0031294
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031208, upper bound: 0.0031294
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048614, 0.0048660
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031375, upper bound: 0.0031802
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031340, upper bound: 0.0031794
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048487, 0.0048433
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027057, upper bound: 0.0026811
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027057, upper bound: 0.0026811
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048403, 0.0048519
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030280, upper bound: 0.0030952
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030276, upper bound: 0.0030963
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048474, 0.0048450
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030218, upper bound: 0.0030406
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030218, upper bound: 0.0030406
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048391, 0.0048537
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0030486
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0030711
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048667, 0.0048581
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031139, upper bound: 0.0030878
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030872, upper bound: 0.0031161
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048667, 0.0048583
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031405, upper bound: 0.0031391
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031595, upper bound: 0.0031424
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048652, 0.0048540
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030976, upper bound: 0.0029813
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030463, upper bound: 0.0030456
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048621, 0.0048572
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031042, upper bound: 0.0031295
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031055, upper bound: 0.0031268
time: 1.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030798, upper bound: 0.0030930
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030798, upper bound: 0.0030917
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030809, upper bound: 0.0031381
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030808, upper bound: 0.0031371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030952, upper bound: 0.0030981
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030897, upper bound: 0.0031021
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031197, upper bound: 0.0031476
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031197, upper bound: 0.0031508
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030164, upper bound: 0.0030456
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030164, upper bound: 0.0030784
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030276, upper bound: 0.0030435
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030325, upper bound: 0.0030461
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031313, upper bound: 0.0031441
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031253, upper bound: 0.0031440
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0027631, upper bound: 0.0027564
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0027631, upper bound: 0.0027564
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031831, upper bound: 0.0031832
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031795, upper bound: 0.0031876
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031262, upper bound: 0.0031337
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031262, upper bound: 0.0031543
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031208, upper bound: 0.0031294
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031208, upper bound: 0.0031294
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031375, upper bound: 0.0031802
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031340, upper bound: 0.0031794
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0027057, upper bound: 0.0026811
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0027057, upper bound: 0.0026811
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030280, upper bound: 0.0030952
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030276, upper bound: 0.0030963
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030218, upper bound: 0.0030406
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030218, upper bound: 0.0030406
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0030486
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0030711
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031139, upper bound: 0.0030878
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030872, upper bound: 0.0031161
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031405, upper bound: 0.0031391
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031595, upper bound: 0.0031424
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030976, upper bound: 0.0029813
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0030463, upper bound: 0.0030456
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031042, upper bound: 0.0031295
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 1, lower bound: -0.0031055, upper bound: 0.0031268

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048536, 0.0048526
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030701, upper bound: 0.0030851
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030701, upper bound: 0.0030884
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048534, 0.0048527
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030701, upper bound: 0.0030833
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030701, upper bound: 0.0030871
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048510, 0.0048575
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030635, upper bound: 0.0031276
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030634, upper bound: 0.0031294
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048507, 0.0048574
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030448, upper bound: 0.0030792
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030390, upper bound: 0.0030846
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048579, 0.0048614
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030730, upper bound: 0.0030419
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030437, upper bound: 0.0030759
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048570, 0.0048624
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030842, upper bound: 0.0031001
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030874, upper bound: 0.0030984
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048579, 0.0048622
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031100, upper bound: 0.0031357
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031100, upper bound: 0.0031357
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048567, 0.0048638
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030891, upper bound: 0.0030943
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030569, upper bound: 0.0031078
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048663, 0.0048580
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030116, upper bound: 0.0030436
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030116, upper bound: 0.0030405
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048631, 0.0048615
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030111, upper bound: 0.0030700
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030111, upper bound: 0.0030750
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048695, 0.0048643
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030321, upper bound: 0.0030412
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030321, upper bound: 0.0030398
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048687, 0.0048654
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029527, upper bound: 0.0029696
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029527, upper bound: 0.0030057
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048584, 0.0048589
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030885, upper bound: 0.0030830
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030824, upper bound: 0.0030892
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048583, 0.0048587
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028178, upper bound: 0.0028143
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028178, upper bound: 0.0028143
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048649, 0.0048654
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031636, upper bound: 0.0031227
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031103, upper bound: 0.0031602
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048641, 0.0048659
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031765, upper bound: 0.0031856
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031777, upper bound: 0.0031851
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048656, 0.0048629
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030930, upper bound: 0.0030778
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030894, upper bound: 0.0030820
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048614, 0.0048673
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030845, upper bound: 0.0030938
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030805, upper bound: 0.0030953
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048625, 0.0048604
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031087, upper bound: 0.0031157
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031087, upper bound: 0.0031184
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048624, 0.0048607
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028815, upper bound: 0.0028551
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028815, upper bound: 0.0028551
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048598, 0.0048643
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030934, upper bound: 0.0031277
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030867, upper bound: 0.0031334
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048597, 0.0048645
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030700, upper bound: 0.0031210
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030700, upper bound: 0.0031403
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048398, 0.0048514
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029979, upper bound: 0.0030321
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029728, upper bound: 0.0030570
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048397, 0.0048519
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029728, upper bound: 0.0030354
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029728, upper bound: 0.0030577
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048469, 0.0048444
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029624, upper bound: 0.0029728
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029624, upper bound: 0.0030175
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048469, 0.0048450
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030171, upper bound: 0.0030382
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030171, upper bound: 0.0030374
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048398, 0.0048501
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023824, upper bound: 0.0023889
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023824, upper bound: 0.0023889
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048354, 0.0048537
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029792, upper bound: 0.0030689
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029881, upper bound: 0.0030674
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048670, 0.0048540
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024485, upper bound: 0.0024460
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024485, upper bound: 0.0024460
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048626, 0.0048581
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031040, upper bound: 0.0031052
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031023, upper bound: 0.0031077
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048669, 0.0048575
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031485, upper bound: 0.0031278
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031264, upper bound: 0.0031285
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048658, 0.0048591
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027577, upper bound: 0.0027576
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027577, upper bound: 0.0027576
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048536, 0.0048339
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029731, upper bound: 0.0029731
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030862, upper bound: 0.0029763
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048452, 0.0048423
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0030285
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0030368
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048597, 0.0048546
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030574, upper bound: 0.0030785
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030538, upper bound: 0.0030847
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048594, 0.0048548
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030632, upper bound: 0.0030637
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030235, upper bound: 0.0030896
time: 1.90 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.21 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030701, upper bound: 0.0030851
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030701, upper bound: 0.0030884
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030701, upper bound: 0.0030833
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030701, upper bound: 0.0030871
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030635, upper bound: 0.0031276
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030634, upper bound: 0.0031294
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030448, upper bound: 0.0030792
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030390, upper bound: 0.0030846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030730, upper bound: 0.0030419
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030437, upper bound: 0.0030759
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030842, upper bound: 0.0031001
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030874, upper bound: 0.0030984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031100, upper bound: 0.0031357
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031100, upper bound: 0.0031357
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030891, upper bound: 0.0030943
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030569, upper bound: 0.0031078
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030116, upper bound: 0.0030436
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030116, upper bound: 0.0030405
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030111, upper bound: 0.0030700
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030111, upper bound: 0.0030750
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030321, upper bound: 0.0030412
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030321, upper bound: 0.0030398
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029527, upper bound: 0.0029696
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029527, upper bound: 0.0030057
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030885, upper bound: 0.0030830
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030824, upper bound: 0.0030892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0028178, upper bound: 0.0028143
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0028178, upper bound: 0.0028143
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031636, upper bound: 0.0031227
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031103, upper bound: 0.0031602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031765, upper bound: 0.0031856
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031777, upper bound: 0.0031851
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030930, upper bound: 0.0030778
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030894, upper bound: 0.0030820
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030845, upper bound: 0.0030938
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030805, upper bound: 0.0030953
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031087, upper bound: 0.0031157
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031087, upper bound: 0.0031184
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0028815, upper bound: 0.0028551
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0028815, upper bound: 0.0028551
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030934, upper bound: 0.0031277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030867, upper bound: 0.0031334
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030700, upper bound: 0.0031210
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030700, upper bound: 0.0031403
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029979, upper bound: 0.0030321
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029728, upper bound: 0.0030570
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029728, upper bound: 0.0030354
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029728, upper bound: 0.0030577
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029624, upper bound: 0.0029728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029624, upper bound: 0.0030175
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030171, upper bound: 0.0030382
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030171, upper bound: 0.0030374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0023824, upper bound: 0.0023889
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0023824, upper bound: 0.0023889
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029792, upper bound: 0.0030689
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029881, upper bound: 0.0030674
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0024485, upper bound: 0.0024460
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0024485, upper bound: 0.0024460
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031040, upper bound: 0.0031052
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031023, upper bound: 0.0031077
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031485, upper bound: 0.0031278
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0031264, upper bound: 0.0031285
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0027577, upper bound: 0.0027576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0027577, upper bound: 0.0027576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029731, upper bound: 0.0029731
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030862, upper bound: 0.0029763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0030285
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0030368
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030574, upper bound: 0.0030785
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030538, upper bound: 0.0030847
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030632, upper bound: 0.0030637
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 1, lower bound: -0.0030235, upper bound: 0.0030896

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048527, 0.0048512
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031008, upper bound: 0.0030703
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030521, upper bound: 0.0030765
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048520, 0.0048518
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030576, upper bound: 0.0030761
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030576, upper bound: 0.0030782
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048525, 0.0048512
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027376, upper bound: 0.0027222
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027376, upper bound: 0.0027222
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048518, 0.0048518
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030521, upper bound: 0.0030722
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030521, upper bound: 0.0030783
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048522, 0.0048568
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027065, upper bound: 0.0027327
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027065, upper bound: 0.0027327
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048503, 0.0048580
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029514, upper bound: 0.0030229
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029514, upper bound: 0.0030751
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048504, 0.0048564
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030005, upper bound: 0.0029989
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029485, upper bound: 0.0030330
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048497, 0.0048574
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030196, upper bound: 0.0030712
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030196, upper bound: 0.0030756
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048540, 0.0048545
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030417, upper bound: 0.0030392
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030712, upper bound: 0.0030400
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048510, 0.0048585
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024914, upper bound: 0.0025020
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024914, upper bound: 0.0025020
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048547, 0.0048598
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030767, upper bound: 0.0030869
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030660, upper bound: 0.0030911
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048545, 0.0048601
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030397, upper bound: 0.0030156
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029964, upper bound: 0.0030544
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048574, 0.0048616
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030734, upper bound: 0.0030739
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0030801
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048573, 0.0048622
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029965, upper bound: 0.0030349
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029965, upper bound: 0.0030848
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048564, 0.0048592
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030779, upper bound: 0.0030823
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030452, upper bound: 0.0030826
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048521, 0.0048638
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030539, upper bound: 0.0031062
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030652, upper bound: 0.0031058
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048638, 0.0048554
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030002, upper bound: 0.0030420
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030002, upper bound: 0.0030409
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048636, 0.0048555
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030050, upper bound: 0.0030325
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030050, upper bound: 0.0030368
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048618, 0.0048596
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029971, upper bound: 0.0030576
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029971, upper bound: 0.0030586
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048609, 0.0048602
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029157, upper bound: 0.0029718
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029157, upper bound: 0.0030283
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048671, 0.0048616
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029726, upper bound: 0.0029887
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029726, upper bound: 0.0030178
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048669, 0.0048618
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030288, upper bound: 0.0030380
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030317, upper bound: 0.0030380
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048564, 0.0048439
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029493, upper bound: 0.0029672
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029847, upper bound: 0.0029663
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048481, 0.0048525
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029453, upper bound: 0.0030041
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029627, upper bound: 0.0030041
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048582, 0.0048578
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030384, upper bound: 0.0030269
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030164, upper bound: 0.0030391
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048573, 0.0048589
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025221, upper bound: 0.0025109
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025221, upper bound: 0.0025109
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048614, 0.0048586
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030024, upper bound: 0.0030096
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030024, upper bound: 0.0030711
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048581, 0.0048616
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030777, upper bound: 0.0031134
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030691, upper bound: 0.0031197
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048624, 0.0048639
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031315, upper bound: 0.0031339
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031220, upper bound: 0.0031406
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048623, 0.0048642
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031143, upper bound: 0.0031401
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031375, upper bound: 0.0031226
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048654, 0.0048618
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030722, upper bound: 0.0030724
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030882, upper bound: 0.0030740
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048645, 0.0048629
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030454, upper bound: 0.0030055
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029990, upper bound: 0.0030371
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048612, 0.0048662
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030825, upper bound: 0.0030920
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030829, upper bound: 0.0030921
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048603, 0.0048673
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030768, upper bound: 0.0030890
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030660, upper bound: 0.0030920
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048620, 0.0048598
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030520, upper bound: 0.0030648
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030520, upper bound: 0.0030594
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048619, 0.0048604
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027687, upper bound: 0.0027424
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027687, upper bound: 0.0027424
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048610, 0.0048612
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027939, upper bound: 0.0027693
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027939, upper bound: 0.0027693
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048624, 0.0048593
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028249, upper bound: 0.0027737
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028061, upper bound: 0.0027989
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048599, 0.0048632
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030797, upper bound: 0.0031155
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030814, upper bound: 0.0031150
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048588, 0.0048643
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030803, upper bound: 0.0031284
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030803, upper bound: 0.0031301
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048600, 0.0048603
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030466, upper bound: 0.0030726
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030495, upper bound: 0.0030605
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048555, 0.0048645
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030734, upper bound: 0.0031331
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030630, upper bound: 0.0031372
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048408, 0.0048477
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023755, upper bound: 0.0023723
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023755, upper bound: 0.0023723
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048362, 0.0048514
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029638, upper bound: 0.0030482
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029638, upper bound: 0.0030535
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048407, 0.0048482
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029642, upper bound: 0.0029851
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029453, upper bound: 0.0029878
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048361, 0.0048519
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029112, upper bound: 0.0029933
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029112, upper bound: 0.0030343
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048436, 0.0048378
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030301, upper bound: 0.0029251
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029112, upper bound: 0.0029431
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048403, 0.0048416
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030229, upper bound: 0.0030154
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030232, upper bound: 0.0030138
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048443, 0.0048422
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030062, upper bound: 0.0030311
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030062, upper bound: 0.0030339
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048443, 0.0048424
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030673, upper bound: 0.0030239
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029999, upper bound: 0.0030239
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048328, 0.0048512
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023748, upper bound: 0.0023832
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023748, upper bound: 0.0023832
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048327, 0.0048511
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023767, upper bound: 0.0023820
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023767, upper bound: 0.0023820
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048629, 0.0048573
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030591, upper bound: 0.0030916
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030925, upper bound: 0.0030932
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048617, 0.0048589
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024249, upper bound: 0.0024343
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024249, upper bound: 0.0024343
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048665, 0.0048569
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030924, upper bound: 0.0030701
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030693, upper bound: 0.0030784
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048664, 0.0048575
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031398, upper bound: 0.0031201
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031320, upper bound: 0.0031243
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048527, 0.0048322
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029619, upper bound: 0.0029565
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029602, upper bound: 0.0029574
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048521, 0.0048331
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0029737
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030838, upper bound: 0.0029728
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048452, 0.0048411
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027277, upper bound: 0.0027138
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027277, upper bound: 0.0027138
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048439, 0.0048430
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030275, upper bound: 0.0030254
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030270, upper bound: 0.0030283
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048595, 0.0048536
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030497, upper bound: 0.0030729
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030529, upper bound: 0.0030748
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048586, 0.0048546
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030429, upper bound: 0.0030786
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030488, upper bound: 0.0030811
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048596, 0.0048507
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024015, upper bound: 0.0024222
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0024015, upper bound: 0.0024222
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048553, 0.0048548
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030104, upper bound: 0.0030737
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030104, upper bound: 0.0030750
time: 1.71 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.29 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0031008, upper bound: 0.0030703
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030521, upper bound: 0.0030765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030576, upper bound: 0.0030761
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030576, upper bound: 0.0030782
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027376, upper bound: 0.0027222
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027376, upper bound: 0.0027222
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030521, upper bound: 0.0030722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030521, upper bound: 0.0030783
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027065, upper bound: 0.0027327
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027065, upper bound: 0.0027327
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029514, upper bound: 0.0030229
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029514, upper bound: 0.0030751
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030005, upper bound: 0.0029989
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029485, upper bound: 0.0030330
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030196, upper bound: 0.0030712
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030196, upper bound: 0.0030756
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030417, upper bound: 0.0030392
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030712, upper bound: 0.0030400
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0024914, upper bound: 0.0025020
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0024914, upper bound: 0.0025020
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030767, upper bound: 0.0030869
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030660, upper bound: 0.0030911
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030397, upper bound: 0.0030156
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029964, upper bound: 0.0030544
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030734, upper bound: 0.0030739
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0030801
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029965, upper bound: 0.0030349
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029965, upper bound: 0.0030848
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030779, upper bound: 0.0030823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030452, upper bound: 0.0030826
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030539, upper bound: 0.0031062
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030652, upper bound: 0.0031058
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030002, upper bound: 0.0030420
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030002, upper bound: 0.0030409
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030050, upper bound: 0.0030325
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030050, upper bound: 0.0030368
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029971, upper bound: 0.0030576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029971, upper bound: 0.0030586
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029157, upper bound: 0.0029718
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029157, upper bound: 0.0030283
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029726, upper bound: 0.0029887
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029726, upper bound: 0.0030178
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030288, upper bound: 0.0030380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030317, upper bound: 0.0030380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029493, upper bound: 0.0029672
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029847, upper bound: 0.0029663
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029453, upper bound: 0.0030041
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029627, upper bound: 0.0030041
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030384, upper bound: 0.0030269
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030164, upper bound: 0.0030391
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0025221, upper bound: 0.0025109
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0025221, upper bound: 0.0025109
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030024, upper bound: 0.0030096
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030024, upper bound: 0.0030711
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030777, upper bound: 0.0031134
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030691, upper bound: 0.0031197
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0031315, upper bound: 0.0031339
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0031220, upper bound: 0.0031406
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0031143, upper bound: 0.0031401
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0031375, upper bound: 0.0031226
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030722, upper bound: 0.0030724
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030882, upper bound: 0.0030740
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030454, upper bound: 0.0030055
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029990, upper bound: 0.0030371
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030825, upper bound: 0.0030920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030829, upper bound: 0.0030921
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030768, upper bound: 0.0030890
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030660, upper bound: 0.0030920
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030520, upper bound: 0.0030648
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030520, upper bound: 0.0030594
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027687, upper bound: 0.0027424
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027687, upper bound: 0.0027424
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027939, upper bound: 0.0027693
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027939, upper bound: 0.0027693
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0028249, upper bound: 0.0027737
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0028061, upper bound: 0.0027989
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030797, upper bound: 0.0031155
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030814, upper bound: 0.0031150
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030803, upper bound: 0.0031284
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030803, upper bound: 0.0031301
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030466, upper bound: 0.0030726
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030495, upper bound: 0.0030605
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030734, upper bound: 0.0031331
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030630, upper bound: 0.0031372
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0023755, upper bound: 0.0023723
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0023755, upper bound: 0.0023723
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029638, upper bound: 0.0030482
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029638, upper bound: 0.0030535
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029642, upper bound: 0.0029851
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029453, upper bound: 0.0029878
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029112, upper bound: 0.0029933
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029112, upper bound: 0.0030343
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030301, upper bound: 0.0029251
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029112, upper bound: 0.0029431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030229, upper bound: 0.0030154
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030232, upper bound: 0.0030138
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030062, upper bound: 0.0030311
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030062, upper bound: 0.0030339
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030673, upper bound: 0.0030239
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029999, upper bound: 0.0030239
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0023748, upper bound: 0.0023832
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0023748, upper bound: 0.0023832
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0023767, upper bound: 0.0023820
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0023767, upper bound: 0.0023820
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030591, upper bound: 0.0030916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030925, upper bound: 0.0030932
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0024249, upper bound: 0.0024343
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0024249, upper bound: 0.0024343
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030924, upper bound: 0.0030701
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030693, upper bound: 0.0030784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0031398, upper bound: 0.0031201
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0031320, upper bound: 0.0031243
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029619, upper bound: 0.0029565
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029602, upper bound: 0.0029574
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0029737
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030838, upper bound: 0.0029728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027277, upper bound: 0.0027138
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0027277, upper bound: 0.0027138
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030275, upper bound: 0.0030254
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030270, upper bound: 0.0030283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030497, upper bound: 0.0030729
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030529, upper bound: 0.0030748
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030429, upper bound: 0.0030786
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030488, upper bound: 0.0030811
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0024015, upper bound: 0.0024222
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0024015, upper bound: 0.0024222
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030104, upper bound: 0.0030737
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.29
Output dim: 1, lower bound: -0.0030104, upper bound: 0.0030750

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048538, 0.0048505
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029423, upper bound: 0.0029656
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029423, upper bound: 0.0030155
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048520, 0.0048515
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030497, upper bound: 0.0030742
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030497, upper bound: 0.0030743
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004423, 0.0021375, -0.0004423, 0.0021375, -0.0025798, 0.0025798
1: 0.9913528, 0.9968160, 0.9913528, 0.9968160, -0.0054632, 0.0054632
2: -0.0082272, -0.0038930, -0.0082272, -0.0038930, -0.0043342, 0.0043342
3: 0.0021347, 0.0053622, 0.0021347, 0.0053622, -0.0032275, 0.0032275
4: 0.0014938, 0.0069207, 0.0014938, 0.0069207, -0.0054269, 0.0054269
5: 0.0023141, 0.0084274, 0.0023141, 0.0084274, -0.0061133, 0.0061133
6: -0.0042317, 0.0016872, -0.0042317, 0.0016872, -0.0059189, 0.0059189
7: -0.0089069, -0.0062919, -0.0089069, -0.0062919, -0.0026150, 0.0026150
8: 0.0033665, 0.0082497, 0.0033665, 0.0082497, -0.0048516, 0.0048512
9: -0.0051543, -0.0014201, -0.0051543, -0.0014201, -0.0037342, 0.0037342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.09 + 596.55 = 600.64 seconds
