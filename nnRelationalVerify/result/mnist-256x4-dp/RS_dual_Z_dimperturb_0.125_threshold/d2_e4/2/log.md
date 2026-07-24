## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0014131413


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986)
1: (-0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299)
2: (0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397)
3: (-0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972)
4: (-0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930)
5: (0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004)
6: (-0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018)
7: (0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074)
8: (-0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184)
9: (-0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.63 + 1.37 = 3.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
time: 0.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 7, lower bound: -0.0032765, upper bound: 0.0032765

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0032728, upper bound: 0.0032728
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033569, 0.0008418, -0.0033569, 0.0008418, -0.0041986, 0.0041986
1: -0.0044389, -0.0023090, -0.0044389, -0.0023090, -0.0021299, 0.0021299
2: 0.0308502, 0.0357899, 0.0308502, 0.0357899, -0.0049397, 0.0049397
3: -0.0030701, 0.0009271, -0.0030701, 0.0009271, -0.0039972, 0.0039972
4: -0.0040113, 0.0003817, -0.0040113, 0.0003817, -0.0043930, 0.0043930
5: 0.0096119, 0.0137123, 0.0096119, 0.0137123, -0.0041004, 0.0041004
6: -0.0057884, -0.0001866, -0.0057884, -0.0001866, -0.0056018, 0.0056018
7: 0.9726887, 0.9774961, 0.9726887, 0.9774961, -0.0048074, 0.0048074
8: -0.0155219, -0.0026034, -0.0155219, -0.0026034, -0.0129184, 0.0129184
9: -0.0024960, 0.0048975, -0.0024960, 0.0048975, -0.0073935, 0.0073935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.76 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.00 + 598.08 = 601.08 seconds
