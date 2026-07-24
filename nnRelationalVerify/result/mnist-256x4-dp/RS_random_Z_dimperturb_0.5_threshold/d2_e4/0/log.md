## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.25792804


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728)
1: (-0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106)
2: (-0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234)
3: (-0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891)
4: (-0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161)
5: (-0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981)
6: (0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845)
7: (-0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032)
8: (-0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757)
9: (-0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 2.26 = 3.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.26 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.26
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.26
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 0.92 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.12 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 0.99 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.39
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.39
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.39
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.39
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.39
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.39
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.39
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.39
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3910905, upper bound: 0.3910905
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3910905, upper bound: 0.3910905
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4023397, upper bound: 0.4023397
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4023397, upper bound: 0.4023397
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4004018, upper bound: 0.4004018
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4004018, upper bound: 0.4004018
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4020525, upper bound: 0.4020525
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4020525, upper bound: 0.4020525
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.3910905, upper bound: 0.3910905
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.3910905, upper bound: 0.3910905
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4023397, upper bound: 0.4023397
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4023397, upper bound: 0.4023397
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4004018, upper bound: 0.4004018
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4004018, upper bound: 0.4004018
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4020525, upper bound: 0.4020525
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4020525, upper bound: 0.4020525
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3811940, upper bound: 0.3811940
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3811940, upper bound: 0.3811940
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4015053, upper bound: 0.4015053
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4015053, upper bound: 0.4015053
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3897897, upper bound: 0.3897897
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3897897, upper bound: 0.3897897
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3872053, upper bound: 0.3872053
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3872053, upper bound: 0.3872053
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3978719, upper bound: 0.3978719
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3978719, upper bound: 0.3978719
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3954736, upper bound: 0.3954736
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3954736, upper bound: 0.3954736
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4014889, upper bound: 0.4014889
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4014889, upper bound: 0.4014889
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4019732, upper bound: 0.4019732
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4019732, upper bound: 0.4019732
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.20 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3811940, upper bound: 0.3811940
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3811940, upper bound: 0.3811940
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4015053, upper bound: 0.4015053
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4015053, upper bound: 0.4015053
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3897897, upper bound: 0.3897897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3897897, upper bound: 0.3897897
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3872053, upper bound: 0.3872053
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3872053, upper bound: 0.3872053
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3978719, upper bound: 0.3978719
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3978719, upper bound: 0.3978719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3954736, upper bound: 0.3954736
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3954736, upper bound: 0.3954736
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4014889, upper bound: 0.4014889
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4014889, upper bound: 0.4014889
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4019732, upper bound: 0.4019732
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4019732, upper bound: 0.4019732
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3853047, upper bound: 0.3853047
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3853047, upper bound: 0.3853047
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798636, upper bound: 0.3798636
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798636, upper bound: 0.3798636
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798624, upper bound: 0.3798624
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798624, upper bound: 0.3798624
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3795226, upper bound: 0.3795226
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3795226, upper bound: 0.3795226
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4015053, upper bound: 0.4015053
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4015053, upper bound: 0.4015053
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3743545, upper bound: 0.3743545
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3743545, upper bound: 0.3743545
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3715739, upper bound: 0.3715739
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3715739, upper bound: 0.3715739
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3872053, upper bound: 0.3872053
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3872053, upper bound: 0.3872053
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3870173, upper bound: 0.3870173
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3870173, upper bound: 0.3870173
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3964893, upper bound: 0.3964893
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3964893, upper bound: 0.3964893
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3978719, upper bound: 0.3978719
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3978719, upper bound: 0.3978719
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3936912, upper bound: 0.3936912
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3936912, upper bound: 0.3936912
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3945090, upper bound: 0.3945090
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3945090, upper bound: 0.3945090
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4014889, upper bound: 0.4014889
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4014889, upper bound: 0.4014889
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4011509, upper bound: 0.4011509
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4011509, upper bound: 0.4011509
time: 2.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3752304, upper bound: 0.3752304
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3752304, upper bound: 0.3752304
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3685187, upper bound: 0.3685187
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3685187, upper bound: 0.3685187
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4021683, upper bound: 0.4021683
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4021683, upper bound: 0.4021683
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
time: 10.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908679, upper bound: 0.3908679
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908679, upper bound: 0.3908679
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025007, upper bound: 0.4025007
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025007, upper bound: 0.4025007
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3897875, upper bound: 0.3897875
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3897875, upper bound: 0.3897875
time: 1.21 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3853047, upper bound: 0.3853047
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3853047, upper bound: 0.3853047
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3798636, upper bound: 0.3798636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3798636, upper bound: 0.3798636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3798624, upper bound: 0.3798624
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3798624, upper bound: 0.3798624
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3795226, upper bound: 0.3795226
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3795226, upper bound: 0.3795226
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4015053, upper bound: 0.4015053
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4015053, upper bound: 0.4015053
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3743545, upper bound: 0.3743545
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3743545, upper bound: 0.3743545
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3715739, upper bound: 0.3715739
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3715739, upper bound: 0.3715739
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3872053, upper bound: 0.3872053
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3872053, upper bound: 0.3872053
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3870173, upper bound: 0.3870173
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3870173, upper bound: 0.3870173
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3964893, upper bound: 0.3964893
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3964893, upper bound: 0.3964893
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3978719, upper bound: 0.3978719
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3978719, upper bound: 0.3978719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3936912, upper bound: 0.3936912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3936912, upper bound: 0.3936912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3945090, upper bound: 0.3945090
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3945090, upper bound: 0.3945090
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4014889, upper bound: 0.4014889
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4014889, upper bound: 0.4014889
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4011509, upper bound: 0.4011509
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4011509, upper bound: 0.4011509
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3752304, upper bound: 0.3752304
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3752304, upper bound: 0.3752304
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3685187, upper bound: 0.3685187
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3685187, upper bound: 0.3685187
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4021683, upper bound: 0.4021683
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4021683, upper bound: 0.4021683
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4022290, upper bound: 0.4022290
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3908679, upper bound: 0.3908679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3908679, upper bound: 0.3908679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4025007, upper bound: 0.4025007
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4025007, upper bound: 0.4025007
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3897875, upper bound: 0.3897875
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.3897875, upper bound: 0.3897875

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3716397, upper bound: 0.3716397
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3716397, upper bound: 0.3716397
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3892623, upper bound: 0.3892623
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3892623, upper bound: 0.3892623
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3816614, upper bound: 0.3816614
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3816614, upper bound: 0.3816614
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3622163, upper bound: 0.3622163
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3622163, upper bound: 0.3622163
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3838990, upper bound: 0.3838990
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3838990, upper bound: 0.3838990
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3836618, upper bound: 0.3836618
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3836618, upper bound: 0.3836618
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798636, upper bound: 0.3798636
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798636, upper bound: 0.3798636
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3783140, upper bound: 0.3783140
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3783140, upper bound: 0.3783140
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3782391, upper bound: 0.3782391
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3782391, upper bound: 0.3782391
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798624, upper bound: 0.3798624
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798624, upper bound: 0.3798624
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3779105, upper bound: 0.3779105
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3779105, upper bound: 0.3779105
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3785775, upper bound: 0.3785775
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3785775, upper bound: 0.3785775
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4015052, upper bound: 0.4015052
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4015052, upper bound: 0.4015052
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3698530, upper bound: 0.3698530
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3698530, upper bound: 0.3698530
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3736036, upper bound: 0.3736036
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3736036, upper bound: 0.3736036
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3743545, upper bound: 0.3743545
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3743545, upper bound: 0.3743545
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3700119, upper bound: 0.3700119
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3700119, upper bound: 0.3700119
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3694949, upper bound: 0.3694949
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3694949, upper bound: 0.3694949
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3870173, upper bound: 0.3870173
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3870173, upper bound: 0.3870173
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3856290, upper bound: 0.3856290
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3856290, upper bound: 0.3856290
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707178, upper bound: 0.3707178
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707178, upper bound: 0.3707178
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3765758, upper bound: 0.3765758
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3765758, upper bound: 0.3765758
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3769096, upper bound: 0.3769096
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3769096, upper bound: 0.3769096
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3958379, upper bound: 0.3958379
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3958379, upper bound: 0.3958379
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789303, upper bound: 0.3789303
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789303, upper bound: 0.3789303
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3975915, upper bound: 0.3975915
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3975915, upper bound: 0.3975915
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3936912, upper bound: 0.3936912
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3936912, upper bound: 0.3936912
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3920236, upper bound: 0.3920236
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3920236, upper bound: 0.3920236
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3778898, upper bound: 0.3778898
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3778898, upper bound: 0.3778898
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3945090, upper bound: 0.3945090
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3945090, upper bound: 0.3945090
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3992893, upper bound: 0.3992893
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3992893, upper bound: 0.3992893
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3790784, upper bound: 0.3790784
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3790784, upper bound: 0.3790784
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3983658, upper bound: 0.3983658
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3983658, upper bound: 0.3983658
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3987006, upper bound: 0.3987006
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3987006, upper bound: 0.3987006
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3526488, upper bound: 0.3526488
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3526488, upper bound: 0.3526488
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3593777, upper bound: 0.3593777
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3593777, upper bound: 0.3593777
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751940, upper bound: 0.3751940
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751940, upper bound: 0.3751940
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3732250, upper bound: 0.3732250
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3732250, upper bound: 0.3732250
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3668768, upper bound: 0.3668768
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3668768, upper bound: 0.3668768
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3634150, upper bound: 0.3634150
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3634150, upper bound: 0.3634150
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3838692, upper bound: 0.3838692
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3838692, upper bound: 0.3838692
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4019300, upper bound: 0.4019300
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4019300, upper bound: 0.4019300
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4008978, upper bound: 0.4008978
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4008978, upper bound: 0.4008978
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3895400, upper bound: 0.3895400
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3895400, upper bound: 0.3895400
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3896464, upper bound: 0.3896464
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3896464, upper bound: 0.3896464
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908679, upper bound: 0.3908679
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3908679, upper bound: 0.3908679
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3994697, upper bound: 0.3994697
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3994697, upper bound: 0.3994697
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4023397, upper bound: 0.4023397
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4023397, upper bound: 0.4023397
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3983551, upper bound: 0.3983551
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3983551, upper bound: 0.3983551
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4019108, upper bound: 0.4019108
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4019108, upper bound: 0.4019108
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3736798, upper bound: 0.3736798
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3736798, upper bound: 0.3736798
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789128, upper bound: 0.3789128
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789128, upper bound: 0.3789128
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789128, upper bound: 0.3789128
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789128, upper bound: 0.3789128
time: 1.11 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3899313, upper bound: 0.3899313
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3716397, upper bound: 0.3716397
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3716397, upper bound: 0.3716397
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3892623, upper bound: 0.3892623
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3892623, upper bound: 0.3892623
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3816614, upper bound: 0.3816614
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3816614, upper bound: 0.3816614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3622163, upper bound: 0.3622163
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3622163, upper bound: 0.3622163
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3838990, upper bound: 0.3838990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3838990, upper bound: 0.3838990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3836618, upper bound: 0.3836618
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3836618, upper bound: 0.3836618
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3798636, upper bound: 0.3798636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3798636, upper bound: 0.3798636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3783140, upper bound: 0.3783140
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3783140, upper bound: 0.3783140
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3782391, upper bound: 0.3782391
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3782391, upper bound: 0.3782391
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3798624, upper bound: 0.3798624
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3798624, upper bound: 0.3798624
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3779105, upper bound: 0.3779105
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3779105, upper bound: 0.3779105
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3785775, upper bound: 0.3785775
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3785775, upper bound: 0.3785775
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4015052, upper bound: 0.4015052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4015052, upper bound: 0.4015052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3698530, upper bound: 0.3698530
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3698530, upper bound: 0.3698530
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3736036, upper bound: 0.3736036
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3736036, upper bound: 0.3736036
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3743545, upper bound: 0.3743545
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3743545, upper bound: 0.3743545
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3700119, upper bound: 0.3700119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3700119, upper bound: 0.3700119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3694949, upper bound: 0.3694949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3694949, upper bound: 0.3694949
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3870173, upper bound: 0.3870173
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3870173, upper bound: 0.3870173
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3856290, upper bound: 0.3856290
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3856290, upper bound: 0.3856290
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3707178, upper bound: 0.3707178
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3707178, upper bound: 0.3707178
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3765758, upper bound: 0.3765758
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3765758, upper bound: 0.3765758
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3769096, upper bound: 0.3769096
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3769096, upper bound: 0.3769096
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3958379, upper bound: 0.3958379
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3958379, upper bound: 0.3958379
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3789303, upper bound: 0.3789303
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3789303, upper bound: 0.3789303
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3975915, upper bound: 0.3975915
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3975915, upper bound: 0.3975915
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3936912, upper bound: 0.3936912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3936912, upper bound: 0.3936912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3920236, upper bound: 0.3920236
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3920236, upper bound: 0.3920236
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3778898, upper bound: 0.3778898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3778898, upper bound: 0.3778898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3945090, upper bound: 0.3945090
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3945090, upper bound: 0.3945090
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3992893, upper bound: 0.3992893
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3992893, upper bound: 0.3992893
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3790784, upper bound: 0.3790784
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3790784, upper bound: 0.3790784
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3983658, upper bound: 0.3983658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3983658, upper bound: 0.3983658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3987006, upper bound: 0.3987006
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3987006, upper bound: 0.3987006
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3526488, upper bound: 0.3526488
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3526488, upper bound: 0.3526488
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3593777, upper bound: 0.3593777
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3593777, upper bound: 0.3593777
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3595473, upper bound: 0.3595473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3751940, upper bound: 0.3751940
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3751940, upper bound: 0.3751940
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3732250, upper bound: 0.3732250
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3732250, upper bound: 0.3732250
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3668768, upper bound: 0.3668768
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3668768, upper bound: 0.3668768
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3634150, upper bound: 0.3634150
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3634150, upper bound: 0.3634150
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3908362, upper bound: 0.3908362
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3838692, upper bound: 0.3838692
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3838692, upper bound: 0.3838692
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4019300, upper bound: 0.4019300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4019300, upper bound: 0.4019300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4008978, upper bound: 0.4008978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4008978, upper bound: 0.4008978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3895400, upper bound: 0.3895400
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3895400, upper bound: 0.3895400
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3896464, upper bound: 0.3896464
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3896464, upper bound: 0.3896464
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3908679, upper bound: 0.3908679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3908679, upper bound: 0.3908679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3994697, upper bound: 0.3994697
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3994697, upper bound: 0.3994697
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4023397, upper bound: 0.4023397
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4023397, upper bound: 0.4023397
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4025308, upper bound: 0.4025308
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3983551, upper bound: 0.3983551
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3983551, upper bound: 0.3983551
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4019108, upper bound: 0.4019108
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4019108, upper bound: 0.4019108
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3736798, upper bound: 0.3736798
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3736798, upper bound: 0.3736798
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.4016867, upper bound: 0.4016867
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3789128, upper bound: 0.3789128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3789128, upper bound: 0.3789128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3789128, upper bound: 0.3789128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 6, lower bound: -0.3789128, upper bound: 0.3789128

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3864813, upper bound: 0.3864813
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3864813, upper bound: 0.3864813
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3837109, upper bound: 0.3837109
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3837109, upper bound: 0.3837109
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3583644, upper bound: 0.3583644
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3583644, upper bound: 0.3583644
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3696370, upper bound: 0.3696370
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3696370, upper bound: 0.3696370
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3846864, upper bound: 0.3846864
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3846864, upper bound: 0.3846864
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3783941, upper bound: 0.3783941
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3783941, upper bound: 0.3783941
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3816614, upper bound: 0.3816614
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3816614, upper bound: 0.3816614
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3791417, upper bound: 0.3791417
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3791417, upper bound: 0.3791417
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3604653, upper bound: 0.3604653
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3604653, upper bound: 0.3604653
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3605692, upper bound: 0.3605692
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3605692, upper bound: 0.3605692
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3875462, upper bound: 0.3875462
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3868803, upper bound: 0.3868803
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3868803, upper bound: 0.3868803
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3583639, upper bound: 0.3583639
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3583639, upper bound: 0.3583639
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2074085, 0.2077643, -0.2074085, 0.2077643, -0.4151728, 0.4151728
1: -0.2157996, 0.2552110, -0.2157996, 0.2552110, -0.4710106, 0.4710106
2: -0.2165362, 0.3425871, -0.2165362, 0.3425871, -0.5591234, 0.5591234
3: -0.1518139, 0.2013751, -0.1518139, 0.2013751, -0.3531891, 0.3531891
4: -0.2800131, 0.2308030, -0.2800131, 0.2308030, -0.5108161, 0.5108161
5: -0.2208213, 0.2969768, -0.2208213, 0.2969768, -0.5177981, 0.5177981
6: 0.5610130, 1.0765976, 0.5610130, 1.0765976, -0.5155845, 0.5155845
7: -0.2948272, 0.2727759, -0.2948272, 0.2727759, -0.5676032, 0.5676032
8: -0.2050808, 0.3153950, -0.2050808, 0.3153950, -0.5204757, 0.5204757
9: -0.2703477, 0.2405492, -0.2703477, 0.2405492, -0.5108969, 0.5108969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.64 + 597.22 = 600.87 seconds
