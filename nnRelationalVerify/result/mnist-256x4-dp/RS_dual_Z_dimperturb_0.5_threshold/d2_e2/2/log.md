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
0: (-0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984)
1: (0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909)
2: (-0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810)
3: (0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257)
4: (0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157)
5: (0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205)
6: (-0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859)
7: (-0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325)
8: (0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043803, 0.0043803)
9: (-0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 2.07 = 3.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0032416, upper bound: 0.0032417

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032401, upper bound: 0.0032397
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032397, upper bound: 0.0032401
time: 0.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.28
Output dim: 1, lower bound: -0.0032401, upper bound: 0.0032397
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.28
Output dim: 1, lower bound: -0.0032397, upper bound: 0.0032401

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043795, 0.0043789
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032358, upper bound: 0.0032295
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032297, upper bound: 0.0032355
time: 1.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043789, 0.0043795
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032355, upper bound: 0.0032297
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032295, upper bound: 0.0032357
time: 1.21 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 1, lower bound: -0.0032358, upper bound: 0.0032295
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 1, lower bound: -0.0032297, upper bound: 0.0032355
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 1, lower bound: -0.0032355, upper bound: 0.0032297
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 1, lower bound: -0.0032295, upper bound: 0.0032357

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043781, 0.0043765
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029346, upper bound: 0.0029286
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029346, upper bound: 0.0029286
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043772, 0.0043774
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029286, upper bound: 0.0029346
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029286, upper bound: 0.0029346
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043774, 0.0043772
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029346, upper bound: 0.0029287
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029346, upper bound: 0.0029287
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043765, 0.0043781
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029286, upper bound: 0.0029346
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029286, upper bound: 0.0029346
time: 0.85 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 1, lower bound: -0.0029346, upper bound: 0.0029286
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 1, lower bound: -0.0029346, upper bound: 0.0029286
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 1, lower bound: -0.0029286, upper bound: 0.0029346
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 1, lower bound: -0.0029286, upper bound: 0.0029346
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 1, lower bound: -0.0029346, upper bound: 0.0029287
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 1, lower bound: -0.0029346, upper bound: 0.0029287
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 1, lower bound: -0.0029286, upper bound: 0.0029346
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 1, lower bound: -0.0029286, upper bound: 0.0029346

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043779, 0.0043760
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029197, upper bound: 0.0029129
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029188, upper bound: 0.0029136
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043776, 0.0043765
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029197, upper bound: 0.0029129
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029188, upper bound: 0.0029136
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043770, 0.0043769
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029136, upper bound: 0.0029189
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029129, upper bound: 0.0029197
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043766, 0.0043774
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029136, upper bound: 0.0029189
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029129, upper bound: 0.0029197
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043773, 0.0043766
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029197, upper bound: 0.0029129
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029189, upper bound: 0.0029136
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043769, 0.0043772
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029197, upper bound: 0.0029129
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029189, upper bound: 0.0029136
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043764, 0.0043776
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029136, upper bound: 0.0029188
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029129, upper bound: 0.0029197
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043760, 0.0043781
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029136, upper bound: 0.0029188
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029129, upper bound: 0.0029197
time: 0.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029197, upper bound: 0.0029129
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029188, upper bound: 0.0029136
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029197, upper bound: 0.0029129
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029188, upper bound: 0.0029136
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029136, upper bound: 0.0029189
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029129, upper bound: 0.0029197
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029136, upper bound: 0.0029189
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029129, upper bound: 0.0029197
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029197, upper bound: 0.0029129
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029189, upper bound: 0.0029136
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029197, upper bound: 0.0029129
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029189, upper bound: 0.0029136
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029136, upper bound: 0.0029188
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029129, upper bound: 0.0029197
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029136, upper bound: 0.0029188
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 1, lower bound: -0.0029129, upper bound: 0.0029197

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043769, 0.0043734
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028959, upper bound: 0.0028698
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028744, upper bound: 0.0028888
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043753, 0.0043750
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028948, upper bound: 0.0028699
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028733, upper bound: 0.0028896
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043767, 0.0043739
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028959, upper bound: 0.0028698
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028744, upper bound: 0.0028888
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043750, 0.0043755
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028948, upper bound: 0.0028699
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028733, upper bound: 0.0028896
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043758, 0.0043743
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028896, upper bound: 0.0028721
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028711, upper bound: 0.0028950
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043744, 0.0043760
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028887, upper bound: 0.0028728
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028706, upper bound: 0.0028959
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043757, 0.0043748
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028896, upper bound: 0.0028721
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028711, upper bound: 0.0028950
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043740, 0.0043765
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028887, upper bound: 0.0028728
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028706, upper bound: 0.0028959
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043762, 0.0043740
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028959, upper bound: 0.0028706
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028728, upper bound: 0.0028887
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043747, 0.0043757
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028950, upper bound: 0.0028711
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028721, upper bound: 0.0028896
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043760, 0.0043745
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028959, upper bound: 0.0028706
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028728, upper bound: 0.0028887
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043743, 0.0043762
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028950, upper bound: 0.0028711
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028721, upper bound: 0.0028896
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043751, 0.0043750
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028896, upper bound: 0.0028733
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028699, upper bound: 0.0028948
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043739, 0.0043767
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028888, upper bound: 0.0028744
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028698, upper bound: 0.0028959
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043750, 0.0043755
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028896, upper bound: 0.0028733
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028699, upper bound: 0.0028948
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043734, 0.0043772
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028888, upper bound: 0.0028744
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028698, upper bound: 0.0028959
time: 1.00 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028959, upper bound: 0.0028698
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028744, upper bound: 0.0028888
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028948, upper bound: 0.0028699
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028733, upper bound: 0.0028896
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028959, upper bound: 0.0028698
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028744, upper bound: 0.0028888
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028948, upper bound: 0.0028699
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028733, upper bound: 0.0028896
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028896, upper bound: 0.0028721
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028711, upper bound: 0.0028950
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028887, upper bound: 0.0028728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028706, upper bound: 0.0028959
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028896, upper bound: 0.0028721
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028711, upper bound: 0.0028950
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028887, upper bound: 0.0028728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028706, upper bound: 0.0028959
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028959, upper bound: 0.0028706
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028728, upper bound: 0.0028887
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028950, upper bound: 0.0028711
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028721, upper bound: 0.0028896
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028959, upper bound: 0.0028706
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028728, upper bound: 0.0028887
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028950, upper bound: 0.0028711
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028721, upper bound: 0.0028896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028896, upper bound: 0.0028733
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028699, upper bound: 0.0028948
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028888, upper bound: 0.0028744
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028698, upper bound: 0.0028959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028896, upper bound: 0.0028733
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028699, upper bound: 0.0028948
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028888, upper bound: 0.0028744
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 1, lower bound: -0.0028698, upper bound: 0.0028959

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043725, 0.0043663
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027711, upper bound: 0.0027517
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027711, upper bound: 0.0027517
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043698, 0.0043691
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027676
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027676
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043711, 0.0043678
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027710, upper bound: 0.0027516
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027710, upper bound: 0.0027516
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043681, 0.0043707
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027677
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027677
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043723, 0.0043668
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027711, upper bound: 0.0027517
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027711, upper bound: 0.0027517
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043696, 0.0043696
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027676
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027676
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043707, 0.0043683
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027710, upper bound: 0.0027516
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027710, upper bound: 0.0027516
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043678, 0.0043712
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027677
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027677
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043715, 0.0043672
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027656, upper bound: 0.0027539
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027656, upper bound: 0.0027539
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043687, 0.0043701
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027729
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027729
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043700, 0.0043689
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027652, upper bound: 0.0027544
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027652, upper bound: 0.0027544
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043673, 0.0043716
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027729
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027729
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043714, 0.0043677
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027656, upper bound: 0.0027539
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027656, upper bound: 0.0027539
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043686, 0.0043706
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027729
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027729
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043697, 0.0043694
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027652, upper bound: 0.0027544
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027652, upper bound: 0.0027544
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043669, 0.0043721
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027729
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027729
time: 2.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043718, 0.0043669
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027517
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027517
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043690, 0.0043697
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027652
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027652
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043705, 0.0043686
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027516
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027516
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043676, 0.0043714
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027656
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027656
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043716, 0.0043674
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027517
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027517
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043689, 0.0043702
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027652
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027652
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043701, 0.0043691
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027516
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027516
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043672, 0.0043719
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027656
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027656
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043707, 0.0043678
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027677, upper bound: 0.0027539
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027677, upper bound: 0.0027539
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043679, 0.0043707
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027710
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027710
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043694, 0.0043696
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027676, upper bound: 0.0027544
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027676, upper bound: 0.0027544
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043667, 0.0043723
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027711
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027711
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043707, 0.0043683
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027677, upper bound: 0.0027539
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027677, upper bound: 0.0027539
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043678, 0.0043712
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027710
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027710
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043691, 0.0043701
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027676, upper bound: 0.0027544
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027676, upper bound: 0.0027544
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0004668, 0.0020317, -0.0004668, 0.0020317, -0.0024984, 0.0024984
1: 0.9915768, 0.9968677, 0.9915768, 0.9968677, -0.0052909, 0.0052909
2: -0.0082394, -0.0042583, -0.0082394, -0.0042583, -0.0039810, 0.0039810
3: 0.0021041, 0.0052298, 0.0021041, 0.0052298, -0.0031257, 0.0031257
4: 0.0017405, 0.0065562, 0.0017405, 0.0065562, -0.0048157, 0.0048157
5: 0.0022562, 0.0081767, 0.0022562, 0.0081767, -0.0059205, 0.0059205
6: -0.0039452, 0.0017408, -0.0039452, 0.0017408, -0.0056859, 0.0056859
7: -0.0087997, -0.0062672, -0.0087997, -0.0062672, -0.0025325, 0.0025325
8: 0.0038465, 0.0082529, 0.0038465, 0.0082529, -0.0043663, 0.0043728
9: -0.0050012, -0.0013848, -0.0050012, -0.0013848, -0.0036164, 0.0036164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027711
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027711
time: 0.97 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027711, upper bound: 0.0027517
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027711, upper bound: 0.0027517
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027676
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027676
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027710, upper bound: 0.0027516
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027710, upper bound: 0.0027516
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027677
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027677
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027711, upper bound: 0.0027517
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027711, upper bound: 0.0027517
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027676
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027676
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027710, upper bound: 0.0027516
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027710, upper bound: 0.0027516
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027677
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027677
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027656, upper bound: 0.0027539
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027656, upper bound: 0.0027539
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027729
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027652, upper bound: 0.0027544
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027652, upper bound: 0.0027544
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027729
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027729
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027656, upper bound: 0.0027539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027656, upper bound: 0.0027539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027729
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027729
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027652, upper bound: 0.0027544
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027652, upper bound: 0.0027544
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027729
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027729
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027517
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027517
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027652
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027652
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027516
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027516
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027656
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027656
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027517
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027517
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027652
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027544, upper bound: 0.0027652
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027516
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027729, upper bound: 0.0027516
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027656
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027539, upper bound: 0.0027656
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027677, upper bound: 0.0027539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027677, upper bound: 0.0027539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027710
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027676, upper bound: 0.0027544
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027676, upper bound: 0.0027544
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027711
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027711
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027677, upper bound: 0.0027539
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027677, upper bound: 0.0027539
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027710
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027516, upper bound: 0.0027710
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027676, upper bound: 0.0027544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027676, upper bound: 0.0027544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027711
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0027517, upper bound: 0.0027711

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.23 + 201.25 = 204.48 seconds
