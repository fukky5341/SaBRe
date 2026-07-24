## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00504846


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737)
1: (0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334)
2: (0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385)
3: (-0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964)
4: (-0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994)
5: (0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702)
6: (-0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080)
7: (-0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460)
8: (0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198)
9: (-0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0125433, 0.0125433)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.70 + 2.48 = 4.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0082446, upper bound: 0.0082446

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0082380, upper bound: 0.0082165
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0082165, upper bound: 0.0082380
time: 1.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 8, lower bound: -0.0082380, upper bound: 0.0082165
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 8, lower bound: -0.0082165, upper bound: 0.0082380

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0125433, 0.0125434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0081381, upper bound: 0.0080458
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0080686, upper bound: 0.0081182
time: 1.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0125434, 0.0125433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079474, upper bound: 0.0080067
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079847, upper bound: 0.0079696
time: 1.43 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.67 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.67
Output dim: 8, lower bound: -0.0081381, upper bound: 0.0080458
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.67
Output dim: 8, lower bound: -0.0080686, upper bound: 0.0081182
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.67
Output dim: 8, lower bound: -0.0079474, upper bound: 0.0080067
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.67
Output dim: 8, lower bound: -0.0079847, upper bound: 0.0079696

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123496, 0.0123749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076022, upper bound: 0.0077493
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078173, upper bound: 0.0074836
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123754, 0.0123497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0080557, upper bound: 0.0081049
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0080557, upper bound: 0.0081027
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0118555, 0.0118917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076883, upper bound: 0.0077723
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076883, upper bound: 0.0077723
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0118918, 0.0118573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0077587, upper bound: 0.0078038
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078294, upper bound: 0.0077344
time: 1.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.41 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 8, lower bound: -0.0076022, upper bound: 0.0077493
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 8, lower bound: -0.0078173, upper bound: 0.0074836
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 8, lower bound: -0.0080557, upper bound: 0.0081049
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 8, lower bound: -0.0080557, upper bound: 0.0081027
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 8, lower bound: -0.0076883, upper bound: 0.0077723
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 8, lower bound: -0.0076883, upper bound: 0.0077723
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 8, lower bound: -0.0077587, upper bound: 0.0078038
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 8, lower bound: -0.0078294, upper bound: 0.0077344

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0111083, 0.0111575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075874, upper bound: 0.0077360
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075881, upper bound: 0.0077360
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0111323, 0.0111397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078083, upper bound: 0.0074424
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0077577, upper bound: 0.0074742
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123520, 0.0123224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0080466, upper bound: 0.0080653
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079971, upper bound: 0.0080961
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123477, 0.0123264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079178, upper bound: 0.0079688
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079178, upper bound: 0.0079688
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116954, 0.0117779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075058, upper bound: 0.0075848
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075058, upper bound: 0.0075848
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0117416, 0.0118917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073313, upper bound: 0.0073902
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073036, upper bound: 0.0074134
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0118039, 0.0116836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076617, upper bound: 0.0076550
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075752, upper bound: 0.0077167
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0117182, 0.0118573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078193, upper bound: 0.0076865
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0077773, upper bound: 0.0077250
time: 1.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0075874, upper bound: 0.0077360
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0075881, upper bound: 0.0077360
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0078083, upper bound: 0.0074424
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0077577, upper bound: 0.0074742
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0080466, upper bound: 0.0080653
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0079971, upper bound: 0.0080961
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0079178, upper bound: 0.0079688
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0079178, upper bound: 0.0079688
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0075058, upper bound: 0.0075848
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0075058, upper bound: 0.0075848
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0073313, upper bound: 0.0073902
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0073036, upper bound: 0.0074134
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0076617, upper bound: 0.0076550
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0075752, upper bound: 0.0077167
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0078193, upper bound: 0.0076865
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.35
Output dim: 8, lower bound: -0.0077773, upper bound: 0.0077250

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110883, 0.0111341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075679, upper bound: 0.0076979
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075502, upper bound: 0.0077119
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110830, 0.0111376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070765, upper bound: 0.0072198
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070765, upper bound: 0.0072198
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0111276, 0.0111595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075779, upper bound: 0.0072892
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076602, upper bound: 0.0072269
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0111521, 0.0111327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075323, upper bound: 0.0073203
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076059, upper bound: 0.0072536
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123420, 0.0123383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0077782, upper bound: 0.0078294
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078026, upper bound: 0.0077931
time: 3.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123679, 0.0123094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0077833, upper bound: 0.0079506
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078495, upper bound: 0.0078715
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123424, 0.0123239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079095, upper bound: 0.0079270
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078692, upper bound: 0.0079607
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123477, 0.0123211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075496, upper bound: 0.0075656
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075138, upper bound: 0.0075800
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116899, 0.0117752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068873, upper bound: 0.0070488
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069378, upper bound: 0.0069809
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116954, 0.0117724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074289, upper bound: 0.0074492
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073802, upper bound: 0.0075079
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0115541, 0.0117200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070275, upper bound: 0.0071592
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070643, upper bound: 0.0070600
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0115758, 0.0117042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072900, upper bound: 0.0074002
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072933, upper bound: 0.0074038
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116092, 0.0115089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056626, upper bound: 0.0056241
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056626, upper bound: 0.0056241
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116344, 0.0114889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072172, upper bound: 0.0072981
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072038, upper bound: 0.0073103
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116997, 0.0118676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074122, upper bound: 0.0074498
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075659, upper bound: 0.0072905
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0117277, 0.0118417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072076, upper bound: 0.0074019
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074558, upper bound: 0.0071563
time: 1.45 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0075679, upper bound: 0.0076979
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0075502, upper bound: 0.0077119
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0070765, upper bound: 0.0072198
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0070765, upper bound: 0.0072198
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0075779, upper bound: 0.0072892
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0076602, upper bound: 0.0072269
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0075323, upper bound: 0.0073203
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0076059, upper bound: 0.0072536
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0077782, upper bound: 0.0078294
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0078026, upper bound: 0.0077931
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0077833, upper bound: 0.0079506
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0078495, upper bound: 0.0078715
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0079095, upper bound: 0.0079270
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0078692, upper bound: 0.0079607
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0075496, upper bound: 0.0075656
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0075138, upper bound: 0.0075800
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0068873, upper bound: 0.0070488
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0069378, upper bound: 0.0069809
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0074289, upper bound: 0.0074492
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0073802, upper bound: 0.0075079
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0070275, upper bound: 0.0071592
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0070643, upper bound: 0.0070600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0072900, upper bound: 0.0074002
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0072933, upper bound: 0.0074038
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0056626, upper bound: 0.0056241
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0056626, upper bound: 0.0056241
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0072172, upper bound: 0.0072981
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0072038, upper bound: 0.0073103
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0074122, upper bound: 0.0074498
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0075659, upper bound: 0.0072905
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0072076, upper bound: 0.0074019
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 8, lower bound: -0.0074558, upper bound: 0.0071563

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110130, 0.0110759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074431, upper bound: 0.0075827
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074401, upper bound: 0.0075826
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110266, 0.0110588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074126, upper bound: 0.0075830
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074126, upper bound: 0.0075830
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110957, 0.0110494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065027, upper bound: 0.0066442
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065027, upper bound: 0.0066442
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109948, 0.0111376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065027, upper bound: 0.0066442
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065027, upper bound: 0.0066442
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110369, 0.0109831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071776, upper bound: 0.0070411
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073546, upper bound: 0.0069128
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109513, 0.0111595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073497, upper bound: 0.0069910
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073921, upper bound: 0.0069128
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110612, 0.0109564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073845, upper bound: 0.0071524
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073946, upper bound: 0.0071524
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109757, 0.0111327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073229, upper bound: 0.0070096
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073631, upper bound: 0.0069641
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116450, 0.0116826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074678, upper bound: 0.0075955
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074873, upper bound: 0.0074887
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116863, 0.0116464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068484, upper bound: 0.0068217
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068484, upper bound: 0.0068217
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0122705, 0.0121240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0077756, upper bound: 0.0077542
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076182, upper bound: 0.0079433
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0121824, 0.0123094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072956, upper bound: 0.0075486
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075458, upper bound: 0.0073212
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123317, 0.0123394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071107, upper bound: 0.0070665
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071107, upper bound: 0.0070665
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123578, 0.0123114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076050, upper bound: 0.0077319
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076367, upper bound: 0.0076947
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0121650, 0.0121604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060480, upper bound: 0.0060653
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060480, upper bound: 0.0060653
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0121819, 0.0121378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075068, upper bound: 0.0073799
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073186, upper bound: 0.0075728
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114547, 0.0115231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068769, upper bound: 0.0070380
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068763, upper bound: 0.0070382
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114377, 0.0115479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069264, upper bound: 0.0069192
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068861, upper bound: 0.0069695
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114922, 0.0115950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074220, upper bound: 0.0072853
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071863, upper bound: 0.0074421
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0115115, 0.0115692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070345, upper bound: 0.0071269
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070439, upper bound: 0.0071212
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110252, 0.0111198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070201, upper bound: 0.0070385
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067777, upper bound: 0.0071515
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109556, 0.0111123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070577, upper bound: 0.0070448
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070548, upper bound: 0.0070543
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0115504, 0.0116748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072094, upper bound: 0.0072575
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071528, upper bound: 0.0073156
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0115489, 0.0116800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069265, upper bound: 0.0070063
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069348, upper bound: 0.0070046
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116438, 0.0114616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056544, upper bound: 0.0055611
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055956, upper bound: 0.0056159
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0115619, 0.0115089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051198, upper bound: 0.0051611
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051979, upper bound: 0.0050849
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114474, 0.0113166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070531, upper bound: 0.0070965
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070531, upper bound: 0.0070965
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114905, 0.0113019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068873, upper bound: 0.0068890
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068873, upper bound: 0.0068890
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114766, 0.0116211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074054, upper bound: 0.0072671
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071786, upper bound: 0.0074431
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114553, 0.0116522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072405, upper bound: 0.0070182
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073512, upper bound: 0.0069915
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0104761, 0.0105950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067877, upper bound: 0.0070083
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067874, upper bound: 0.0070238
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0104903, 0.0105841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072275, upper bound: 0.0069538
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072275, upper bound: 0.0069538
time: 1.68 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0074431, upper bound: 0.0075827
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0074401, upper bound: 0.0075826
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0074126, upper bound: 0.0075830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0074126, upper bound: 0.0075830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0065027, upper bound: 0.0066442
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0065027, upper bound: 0.0066442
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0065027, upper bound: 0.0066442
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0065027, upper bound: 0.0066442
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0071776, upper bound: 0.0070411
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0073546, upper bound: 0.0069128
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0073497, upper bound: 0.0069910
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0073921, upper bound: 0.0069128
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0073845, upper bound: 0.0071524
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0073946, upper bound: 0.0071524
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0073229, upper bound: 0.0070096
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0073631, upper bound: 0.0069641
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0074678, upper bound: 0.0075955
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0074873, upper bound: 0.0074887
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0068484, upper bound: 0.0068217
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0068484, upper bound: 0.0068217
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0077756, upper bound: 0.0077542
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0076182, upper bound: 0.0079433
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0072956, upper bound: 0.0075486
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0075458, upper bound: 0.0073212
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0071107, upper bound: 0.0070665
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0071107, upper bound: 0.0070665
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0076050, upper bound: 0.0077319
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0076367, upper bound: 0.0076947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0060480, upper bound: 0.0060653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0060480, upper bound: 0.0060653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0075068, upper bound: 0.0073799
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0073186, upper bound: 0.0075728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0068769, upper bound: 0.0070380
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0068763, upper bound: 0.0070382
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0069264, upper bound: 0.0069192
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0068861, upper bound: 0.0069695
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0074220, upper bound: 0.0072853
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0071863, upper bound: 0.0074421
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0070345, upper bound: 0.0071269
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0070439, upper bound: 0.0071212
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0070201, upper bound: 0.0070385
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0067777, upper bound: 0.0071515
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0070577, upper bound: 0.0070448
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0070548, upper bound: 0.0070543
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0072094, upper bound: 0.0072575
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0071528, upper bound: 0.0073156
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0069265, upper bound: 0.0070063
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0069348, upper bound: 0.0070046
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0056544, upper bound: 0.0055611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0055956, upper bound: 0.0056159
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0051198, upper bound: 0.0051611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0051979, upper bound: 0.0050849
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0070531, upper bound: 0.0070965
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0070531, upper bound: 0.0070965
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0068873, upper bound: 0.0068890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0068873, upper bound: 0.0068890
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0074054, upper bound: 0.0072671
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0071786, upper bound: 0.0074431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0072405, upper bound: 0.0070182
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0073512, upper bound: 0.0069915
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0067877, upper bound: 0.0070083
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0067874, upper bound: 0.0070238
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0072275, upper bound: 0.0069538
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -0.0072275, upper bound: 0.0069538

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109482, 0.0110158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074354, upper bound: 0.0075487
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073879, upper bound: 0.0075749
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109530, 0.0110014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074325, upper bound: 0.0075455
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073879, upper bound: 0.0075748
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110208, 0.0110558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057141, upper bound: 0.0057836
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057141, upper bound: 0.0057836
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110266, 0.0110531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071326, upper bound: 0.0073439
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071326, upper bound: 0.0073414
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109165, 0.0108922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060995, upper bound: 0.0062692
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061268, upper bound: 0.0062090
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109417, 0.0108703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064934, upper bound: 0.0065757
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064279, upper bound: 0.0066350
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0108157, 0.0109803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061906, upper bound: 0.0063808
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062569, upper bound: 0.0063406
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0108467, 0.0109584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061751, upper bound: 0.0063020
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061751, upper bound: 0.0063020
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0108128, 0.0107360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071583, upper bound: 0.0070185
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071520, upper bound: 0.0070186
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0107898, 0.0107842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073357, upper bound: 0.0068906
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073314, upper bound: 0.0068907
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0101839, 0.0103909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0067675
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071091, upper bound: 0.0066889
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0101826, 0.0103794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073845, upper bound: 0.0066405
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072451, upper bound: 0.0069060
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109926, 0.0108978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070908, upper bound: 0.0069099
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071293, upper bound: 0.0068570
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110026, 0.0108932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069982, upper bound: 0.0067314
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069949, upper bound: 0.0067420
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0102459, 0.0104321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073057, upper bound: 0.0069918
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072959, upper bound: 0.0069918
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0102804, 0.0104051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070341, upper bound: 0.0066483
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070341, upper bound: 0.0066483
time: 2.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0111380, 0.0110953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069882, upper bound: 0.0070671
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069882, upper bound: 0.0070671
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110577, 0.0110967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072708, upper bound: 0.0072692
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072709, upper bound: 0.0072692
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0117298, 0.0115997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060485, upper bound: 0.0060552
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060485, upper bound: 0.0060552
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116396, 0.0116464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062803, upper bound: 0.0064297
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064611, upper bound: 0.0062690
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0123658, 0.0124478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074993, upper bound: 0.0075476
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075293, upper bound: 0.0074391
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0125969, 0.0122192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073685, upper bound: 0.0076322
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073685, upper bound: 0.0076322
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109602, 0.0110987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071018, upper bound: 0.0073911
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071018, upper bound: 0.0073911
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0109812, 0.0110850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071171, upper bound: 0.0070792
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073079, upper bound: 0.0069469
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0122955, 0.0122504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065665, upper bound: 0.0066808
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067391, upper bound: 0.0065306
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0122427, 0.0123394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065535, upper bound: 0.0065176
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065535, upper bound: 0.0065176
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116616, 0.0116560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073414, upper bound: 0.0074655
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073414, upper bound: 0.0074655
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0117024, 0.0116196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073137, upper bound: 0.0074176
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073863, upper bound: 0.0073802
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0121292, 0.0120706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052296, upper bound: 0.0052537
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052296, upper bound: 0.0052537
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0120752, 0.0121604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052296, upper bound: 0.0052537
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052296, upper bound: 0.0052537
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0122932, 0.0124786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060400, upper bound: 0.0059880
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060400, upper bound: 0.0059880
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0125110, 0.0122494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069606, upper bound: 0.0073168
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070996, upper bound: 0.0072583
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114289, 0.0114923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065087, upper bound: 0.0066117
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065099, upper bound: 0.0066116
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114242, 0.0114973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063276, upper bound: 0.0066801
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065056, upper bound: 0.0064767
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114263, 0.0115606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065301, upper bound: 0.0065517
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065307, upper bound: 0.0065517
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114504, 0.0115349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068734, upper bound: 0.0069572
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068755, upper bound: 0.0069591
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0115431, 0.0118758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074163, upper bound: 0.0072696
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074068, upper bound: 0.0072770
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0117586, 0.0116461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065571, upper bound: 0.0069165
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066501, upper bound: 0.0068582
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114553, 0.0113980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064231, upper bound: 0.0067846
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067033, upper bound: 0.0065437
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0113403, 0.0115692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070367, upper bound: 0.0069748
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068212, upper bound: 0.0071138
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0110331, 0.0113579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066306, upper bound: 0.0066651
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066404, upper bound: 0.0066637
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0112034, 0.0111299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066209, upper bound: 0.0069896
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066209, upper bound: 0.0069896
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0108747, 0.0110474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067057, upper bound: 0.0067053
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067057, upper bound: 0.0067053
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0108902, 0.0110313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065479, upper bound: 0.0066763
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066514, upper bound: 0.0065916
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0113488, 0.0115106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068417, upper bound: 0.0068859
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068518, upper bound: 0.0068785
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0113708, 0.0114840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065484, upper bound: 0.0069873
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068482, upper bound: 0.0067223
time: 2.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0115042, 0.0115058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063302, upper bound: 0.0064654
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063801, upper bound: 0.0064133
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0113814, 0.0116800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066354, upper bound: 0.0067878
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066820, upper bound: 0.0066773
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0116878, 0.0117211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053505, upper bound: 0.0052784
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053705, upper bound: 0.0052555
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0119023, 0.0115056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055880, upper bound: 0.0056009
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055850, upper bound: 0.0056080
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0103084, 0.0102448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051119, upper bound: 0.0051431
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051100, upper bound: 0.0051532
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0102968, 0.0102366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051896, upper bound: 0.0050196
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051410, upper bound: 0.0050767
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0113748, 0.0112543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070196, upper bound: 0.0070574
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070196, upper bound: 0.0070599
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0113850, 0.0112426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070465, upper bound: 0.0068753
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068821, upper bound: 0.0070896
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114849, 0.0112997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063204, upper bound: 0.0065498
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065511, upper bound: 0.0063193
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0114905, 0.0112963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068219, upper bound: 0.0068241
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068219, upper bound: 0.0068241
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0115277, 0.0118945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073189, upper bound: 0.0071123
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072612, upper bound: 0.0071798
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0117541, 0.0116765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067393, upper bound: 0.0069783
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067390, upper bound: 0.0069789
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037769, 0.0104506, 0.0037769, 0.0104506, -0.0066737, 0.0066737
1: 0.0018680, 0.0028014, 0.0018680, 0.0028014, -0.0009334, 0.0009334
2: 0.0084332, 0.0122717, 0.0084332, 0.0122717, -0.0038385, 0.0038385
3: -0.0057849, -0.0019885, -0.0057849, -0.0019885, -0.0037964, 0.0037964
4: -0.0018843, 0.0021151, -0.0018843, 0.0021151, -0.0039994, 0.0039994
5: 0.0020263, 0.0058965, 0.0020263, 0.0058965, -0.0038702, 0.0038702
6: -0.0140126, 0.0010954, -0.0140126, 0.0010954, -0.0151080, 0.0151080
7: -0.0040485, 0.0167975, -0.0040485, 0.0167975, -0.0208460, 0.0208460
8: 0.9863620, 1.0009818, 0.9863620, 1.0009818, -0.0146198, 0.0146198
9: -0.0165849, -0.0035076, -0.0165849, -0.0035076, -0.0108646, 0.0110599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071496, upper bound: 0.0068734
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070912, upper bound: 0.0069217
time: 1.68 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0074354, upper bound: 0.0075487
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073879, upper bound: 0.0075749
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0074325, upper bound: 0.0075455
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073879, upper bound: 0.0075748
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0057141, upper bound: 0.0057836
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0057141, upper bound: 0.0057836
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071326, upper bound: 0.0073439
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071326, upper bound: 0.0073414
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0060995, upper bound: 0.0062692
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0061268, upper bound: 0.0062090
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0064934, upper bound: 0.0065757
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0064279, upper bound: 0.0066350
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0061906, upper bound: 0.0063808
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0062569, upper bound: 0.0063406
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0061751, upper bound: 0.0063020
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0061751, upper bound: 0.0063020
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071583, upper bound: 0.0070185
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071520, upper bound: 0.0070186
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073357, upper bound: 0.0068906
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073314, upper bound: 0.0068907
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0067675
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071091, upper bound: 0.0066889
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073845, upper bound: 0.0066405
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0072451, upper bound: 0.0069060
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070908, upper bound: 0.0069099
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071293, upper bound: 0.0068570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0069982, upper bound: 0.0067314
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0069949, upper bound: 0.0067420
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073057, upper bound: 0.0069918
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0072959, upper bound: 0.0069918
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070341, upper bound: 0.0066483
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070341, upper bound: 0.0066483
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0069882, upper bound: 0.0070671
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0069882, upper bound: 0.0070671
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0072708, upper bound: 0.0072692
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0072709, upper bound: 0.0072692
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0060485, upper bound: 0.0060552
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0060485, upper bound: 0.0060552
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0062803, upper bound: 0.0064297
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0064611, upper bound: 0.0062690
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0074993, upper bound: 0.0075476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0075293, upper bound: 0.0074391
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073685, upper bound: 0.0076322
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073685, upper bound: 0.0076322
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071018, upper bound: 0.0073911
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071018, upper bound: 0.0073911
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071171, upper bound: 0.0070792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073079, upper bound: 0.0069469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065665, upper bound: 0.0066808
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0067391, upper bound: 0.0065306
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065535, upper bound: 0.0065176
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065535, upper bound: 0.0065176
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073414, upper bound: 0.0074655
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073414, upper bound: 0.0074655
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073137, upper bound: 0.0074176
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073863, upper bound: 0.0073802
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0052296, upper bound: 0.0052537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0052296, upper bound: 0.0052537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0052296, upper bound: 0.0052537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0052296, upper bound: 0.0052537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0060400, upper bound: 0.0059880
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0060400, upper bound: 0.0059880
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0069606, upper bound: 0.0073168
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070996, upper bound: 0.0072583
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065087, upper bound: 0.0066117
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065099, upper bound: 0.0066116
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0063276, upper bound: 0.0066801
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065056, upper bound: 0.0064767
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065301, upper bound: 0.0065517
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065307, upper bound: 0.0065517
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0068734, upper bound: 0.0069572
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0068755, upper bound: 0.0069591
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0074163, upper bound: 0.0072696
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0074068, upper bound: 0.0072770
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065571, upper bound: 0.0069165
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0066501, upper bound: 0.0068582
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0064231, upper bound: 0.0067846
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0067033, upper bound: 0.0065437
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070367, upper bound: 0.0069748
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0068212, upper bound: 0.0071138
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0066306, upper bound: 0.0066651
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0066404, upper bound: 0.0066637
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0066209, upper bound: 0.0069896
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0066209, upper bound: 0.0069896
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0067057, upper bound: 0.0067053
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0067057, upper bound: 0.0067053
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065479, upper bound: 0.0066763
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0066514, upper bound: 0.0065916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0068417, upper bound: 0.0068859
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0068518, upper bound: 0.0068785
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065484, upper bound: 0.0069873
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0068482, upper bound: 0.0067223
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0063302, upper bound: 0.0064654
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0063801, upper bound: 0.0064133
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0066354, upper bound: 0.0067878
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0066820, upper bound: 0.0066773
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0053505, upper bound: 0.0052784
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0053705, upper bound: 0.0052555
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0055880, upper bound: 0.0056009
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0055850, upper bound: 0.0056080
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0051119, upper bound: 0.0051431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0051100, upper bound: 0.0051532
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0051896, upper bound: 0.0050196
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0051410, upper bound: 0.0050767
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070196, upper bound: 0.0070574
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070196, upper bound: 0.0070599
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070465, upper bound: 0.0068753
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0068821, upper bound: 0.0070896
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0063204, upper bound: 0.0065498
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0065511, upper bound: 0.0063193
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0068219, upper bound: 0.0068241
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0068219, upper bound: 0.0068241
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0073189, upper bound: 0.0071123
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0072612, upper bound: 0.0071798
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0067393, upper bound: 0.0069783
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0067390, upper bound: 0.0069789
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0071496, upper bound: 0.0068734
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -0.0070912, upper bound: 0.0069217
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -0.0073512, upper bound: 0.0069915
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -0.0067877, upper bound: 0.0070083
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -0.0067874, upper bound: 0.0070238
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -0.0072275, upper bound: 0.0069538
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -0.0072275, upper bound: 0.0069538

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.18 + 597.00 = 601.18 seconds
