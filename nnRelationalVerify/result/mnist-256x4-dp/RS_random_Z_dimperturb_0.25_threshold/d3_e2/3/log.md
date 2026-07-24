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
Threshold: 0.06656274


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755)
1: (-0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764)
2: (0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182)
3: (0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0485770, 0.0485770)
4: (-0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715)
5: (-0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811)
6: (-0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148)
7: (0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930)
8: (-0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235)
9: (-0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.46 = 3.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0712859, upper bound: 0.0712859

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0709485, upper bound: 0.0705630
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0705630, upper bound: 0.0709485
time: 0.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 7, lower bound: -0.0709485, upper bound: 0.0705630
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 7, lower bound: -0.0705630, upper bound: 0.0709485

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0484211, 0.0485132
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0706859, upper bound: 0.0702824
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0706818, upper bound: 0.0702987
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0485132, 0.0484211
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0702987, upper bound: 0.0706818
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0702824, upper bound: 0.0706859
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 7, lower bound: -0.0706859, upper bound: 0.0702824
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 7, lower bound: -0.0706818, upper bound: 0.0702987
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 7, lower bound: -0.0702987, upper bound: 0.0706818
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 7, lower bound: -0.0702824, upper bound: 0.0706859

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0482101, 0.0482977
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0695415, upper bound: 0.0692667
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0697085, upper bound: 0.0691043
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0482056, 0.0482985
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0695428, upper bound: 0.0693221
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0696788, upper bound: 0.0691043
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0482986, 0.0482056
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0691043, upper bound: 0.0696788
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0693221, upper bound: 0.0695428
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0482977, 0.0482101
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0701361, upper bound: 0.0706622
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0702584, upper bound: 0.0706155
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0695415, upper bound: 0.0692667
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0697085, upper bound: 0.0691043
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0695428, upper bound: 0.0693221
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0696788, upper bound: 0.0691043
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0691043, upper bound: 0.0696788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0693221, upper bound: 0.0695428
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0701361, upper bound: 0.0706622
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0702584, upper bound: 0.0706155

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0479013, 0.0478827
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0694676, upper bound: 0.0692428
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0695186, upper bound: 0.0691423
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477951, 0.0482977
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0696199, upper bound: 0.0690806
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0696844, upper bound: 0.0689304
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478950, 0.0478835
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0694721, upper bound: 0.0692978
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0695198, upper bound: 0.0691465
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477906, 0.0482985
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0696035, upper bound: 0.0690807
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0696554, upper bound: 0.0689271
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0479693, 0.0477906
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0689271, upper bound: 0.0696554
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0690807, upper bound: 0.0696035
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478835, 0.0482056
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0691465, upper bound: 0.0695198
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0692978, upper bound: 0.0694721
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0482562, 0.0481765
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0689304, upper bound: 0.0696844
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0691423, upper bound: 0.0695186
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0482641, 0.0481696
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0690806, upper bound: 0.0696199
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0692428, upper bound: 0.0694676
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0694676, upper bound: 0.0692428
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0695186, upper bound: 0.0691423
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0696199, upper bound: 0.0690806
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0696844, upper bound: 0.0689304
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0694721, upper bound: 0.0692978
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0695198, upper bound: 0.0691465
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0696035, upper bound: 0.0690807
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0696554, upper bound: 0.0689271
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0689271, upper bound: 0.0696554
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0690807, upper bound: 0.0696035
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0691465, upper bound: 0.0695198
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0692978, upper bound: 0.0694721
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0689304, upper bound: 0.0696844
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0691423, upper bound: 0.0695186
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0690806, upper bound: 0.0696199
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 7, lower bound: -0.0692428, upper bound: 0.0694676

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478618, 0.0478509
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675795, upper bound: 0.0678499
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0681634, upper bound: 0.0677115
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478695, 0.0478431
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0662259, upper bound: 0.0664831
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0668818, upper bound: 0.0659120
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477565, 0.0482641
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0695854, upper bound: 0.0660406
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669780, upper bound: 0.0690458
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477633, 0.0482562
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0662259, upper bound: 0.0664413
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669889, upper bound: 0.0658996
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478562, 0.0478517
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666400, upper bound: 0.0664912
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670093, upper bound: 0.0654299
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478632, 0.0478463
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677688, upper bound: 0.0677521
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0681775, upper bound: 0.0675561
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477512, 0.0482649
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0674845, upper bound: 0.0681496
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0686388, upper bound: 0.0668013
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477588, 0.0482595
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0661645, upper bound: 0.0686729
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0694234, upper bound: 0.0659248
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0479310, 0.0477588
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0687902, upper bound: 0.0684703
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678905, upper bound: 0.0695222
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0479375, 0.0477512
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0654162, upper bound: 0.0670928
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0664146, upper bound: 0.0666483
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478464, 0.0481719
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0690132, upper bound: 0.0684156
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680893, upper bound: 0.0693834
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478517, 0.0481643
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0691645, upper bound: 0.0684387
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680896, upper bound: 0.0693351
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0479216, 0.0477633
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672893, upper bound: 0.0682908
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675889, upper bound: 0.0681155
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478431, 0.0481765
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0664752, upper bound: 0.0671024
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666721, upper bound: 0.0669936
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0479277, 0.0477565
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658360, upper bound: 0.0693761
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0688304, upper bound: 0.0664072
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478509, 0.0481696
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0611882, upper bound: 0.0687289
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0684662, upper bound: 0.0605995
time: 0.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0675795, upper bound: 0.0678499
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0681634, upper bound: 0.0677115
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0662259, upper bound: 0.0664831
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0668818, upper bound: 0.0659120
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0695854, upper bound: 0.0660406
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0669780, upper bound: 0.0690458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0662259, upper bound: 0.0664413
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0669889, upper bound: 0.0658996
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0666400, upper bound: 0.0664912
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0670093, upper bound: 0.0654299
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0677688, upper bound: 0.0677521
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0681775, upper bound: 0.0675561
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0674845, upper bound: 0.0681496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0686388, upper bound: 0.0668013
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0661645, upper bound: 0.0686729
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0694234, upper bound: 0.0659248
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0687902, upper bound: 0.0684703
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0678905, upper bound: 0.0695222
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0654162, upper bound: 0.0670928
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0664146, upper bound: 0.0666483
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0690132, upper bound: 0.0684156
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0680893, upper bound: 0.0693834
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0691645, upper bound: 0.0684387
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0680896, upper bound: 0.0693351
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0672893, upper bound: 0.0682908
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0675889, upper bound: 0.0681155
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0664752, upper bound: 0.0671024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0666721, upper bound: 0.0669936
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0658360, upper bound: 0.0693761
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0688304, upper bound: 0.0664072
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0611882, upper bound: 0.0687289
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 7, lower bound: -0.0684662, upper bound: 0.0605995

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0473951, 0.0473360
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636118, upper bound: 0.0633929
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636163, upper bound: 0.0633893
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0473469, 0.0478509
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0681326, upper bound: 0.0634253
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0655620, upper bound: 0.0676801
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476352, 0.0478431
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0668487, upper bound: 0.0630757
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0638625, upper bound: 0.0658782
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476730, 0.0482692
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0661964, upper bound: 0.0635859
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669426, upper bound: 0.0630646
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477629, 0.0481740
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647139, upper bound: 0.0665895
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0647300, upper bound: 0.0664383
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0475289, 0.0482562
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0639202, upper bound: 0.0646910
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0657467, upper bound: 0.0630045
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477192, 0.0476174
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0590598, upper bound: 0.0659597
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0660101, upper bound: 0.0588837
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476219, 0.0478517
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0651159, upper bound: 0.0651532
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667149, upper bound: 0.0634610
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0474043, 0.0473315
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0635008, upper bound: 0.0674505
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0674898, upper bound: 0.0643789
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0473483, 0.0478463
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0587347, upper bound: 0.0618023
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0622173, upper bound: 0.0569338
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0475241, 0.0473695
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0650635, upper bound: 0.0657651
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0651201, upper bound: 0.0656644
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0468411, 0.0479456
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0686062, upper bound: 0.0629719
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0652368, upper bound: 0.0667689
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0461258, 0.0460174
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0661271, upper bound: 0.0665072
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0639445, upper bound: 0.0686360
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0455411, 0.0467667
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678303, upper bound: 0.0644044
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679566, upper bound: 0.0631684
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477951, 0.0476596
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0657669, upper bound: 0.0682083
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0685287, upper bound: 0.0658809
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478311, 0.0476229
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0655222, upper bound: 0.0692807
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676335, upper bound: 0.0660055
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477781, 0.0475168
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0624066, upper bound: 0.0658328
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0642702, upper bound: 0.0639716
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477031, 0.0477512
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0588696, upper bound: 0.0641052
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0634323, upper bound: 0.0596948
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477104, 0.0480771
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0671692, upper bound: 0.0675193
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680476, upper bound: 0.0659460
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477497, 0.0480370
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658779, upper bound: 0.0668350
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0659411, upper bound: 0.0666376
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477158, 0.0480690
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673950, upper bound: 0.0675368
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0681728, upper bound: 0.0659437
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477551, 0.0480293
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0653098, upper bound: 0.0690757
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678378, upper bound: 0.0662820
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0474940, 0.0472484
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0581166, upper bound: 0.0619168
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0620671, upper bound: 0.0584057
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0474067, 0.0477633
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0591140, upper bound: 0.0673240
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0668236, upper bound: 0.0582250
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476280, 0.0478877
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0561900, upper bound: 0.0593836
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0587770, upper bound: 0.0567090
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0475575, 0.0481765
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0648973, upper bound: 0.0660568
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0657245, upper bound: 0.0650335
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0463980, 0.0455388
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0634733, upper bound: 0.0668732
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0635093, upper bound: 0.0667469
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0457101, 0.0461259
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0664958, upper bound: 0.0654917
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678314, upper bound: 0.0617654
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0448987, 0.0438525
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0594323, upper bound: 0.0684877
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0609465, upper bound: 0.0669638
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0435494, 0.0451103
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0684300, upper bound: 0.0600871
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0652333, upper bound: 0.0605571
time: 0.63 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0636118, upper bound: 0.0633929
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0636163, upper bound: 0.0633893
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0681326, upper bound: 0.0634253
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0655620, upper bound: 0.0676801
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0668487, upper bound: 0.0630757
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0638625, upper bound: 0.0658782
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0661964, upper bound: 0.0635859
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0669426, upper bound: 0.0630646
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0647139, upper bound: 0.0665895
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0647300, upper bound: 0.0664383
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0639202, upper bound: 0.0646910
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0657467, upper bound: 0.0630045
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0590598, upper bound: 0.0659597
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0660101, upper bound: 0.0588837
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0651159, upper bound: 0.0651532
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0667149, upper bound: 0.0634610
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0635008, upper bound: 0.0674505
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0674898, upper bound: 0.0643789
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0587347, upper bound: 0.0618023
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0622173, upper bound: 0.0569338
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0650635, upper bound: 0.0657651
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0651201, upper bound: 0.0656644
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0686062, upper bound: 0.0629719
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0652368, upper bound: 0.0667689
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0661271, upper bound: 0.0665072
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0639445, upper bound: 0.0686360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0678303, upper bound: 0.0644044
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0679566, upper bound: 0.0631684
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0657669, upper bound: 0.0682083
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0685287, upper bound: 0.0658809
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0655222, upper bound: 0.0692807
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0676335, upper bound: 0.0660055
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0624066, upper bound: 0.0658328
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0642702, upper bound: 0.0639716
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0588696, upper bound: 0.0641052
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0634323, upper bound: 0.0596948
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0671692, upper bound: 0.0675193
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0680476, upper bound: 0.0659460
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0658779, upper bound: 0.0668350
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0659411, upper bound: 0.0666376
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0673950, upper bound: 0.0675368
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0681728, upper bound: 0.0659437
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0653098, upper bound: 0.0690757
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0678378, upper bound: 0.0662820
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0581166, upper bound: 0.0619168
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0620671, upper bound: 0.0584057
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0591140, upper bound: 0.0673240
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0668236, upper bound: 0.0582250
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0561900, upper bound: 0.0593836
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0587770, upper bound: 0.0567090
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0648973, upper bound: 0.0660568
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0657245, upper bound: 0.0650335
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0634733, upper bound: 0.0668732
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0635093, upper bound: 0.0667469
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0664958, upper bound: 0.0654917
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0678314, upper bound: 0.0617654
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0594323, upper bound: 0.0684877
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0609465, upper bound: 0.0669638
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0684300, upper bound: 0.0600871
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 7, lower bound: -0.0652333, upper bound: 0.0605571

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477784, 0.0478583
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0638442, upper bound: 0.0601145
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0642506, upper bound: 0.0597270
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478634, 0.0477675
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0606473, upper bound: 0.0638590
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0607891, upper bound: 0.0633788
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477861, 0.0478533
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0594317, upper bound: 0.0613762
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640078, upper bound: 0.0586180
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0475221, 0.0482641
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0647085, upper bound: 0.0627792
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666545, upper bound: 0.0619037
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0475584, 0.0479753
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0563710, upper bound: 0.0587291
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0586971, upper bound: 0.0562493
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476006, 0.0477152
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0648579, upper bound: 0.0631934
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0665159, upper bound: 0.0629815
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0463218, 0.0456287
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0591080, upper bound: 0.0667014
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0625044, upper bound: 0.0579186
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0456455, 0.0462164
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673543, upper bound: 0.0640645
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0663766, upper bound: 0.0642426
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476677, 0.0482760
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0654576, upper bound: 0.0626927
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682986, upper bound: 0.0583959
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477552, 0.0481748
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0650535, upper bound: 0.0657627
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650813, upper bound: 0.0666344
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477617, 0.0481695
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615007, upper bound: 0.0662932
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615508, upper bound: 0.0661725
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0473449, 0.0477565
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676988, upper bound: 0.0640645
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0665282, upper bound: 0.0642662
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0472439, 0.0482595
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0594406, upper bound: 0.0622268
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0671870, upper bound: 0.0572501
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0463787, 0.0455411
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630192, upper bound: 0.0668411
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0642662, upper bound: 0.0665282
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0457133, 0.0461258
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0650087, upper bound: 0.0645937
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0660170, upper bound: 0.0645937
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0463787, 0.0455411
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0596782, upper bound: 0.0685669
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0646494, upper bound: 0.0613416
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0457133, 0.0461258
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0592594, upper bound: 0.0650960
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669554, upper bound: 0.0613423
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476822, 0.0472765
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0648323, upper bound: 0.0652187
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0648495, upper bound: 0.0651141
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0469363, 0.0477889
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680146, upper bound: 0.0608458
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0659298, upper bound: 0.0659123
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476749, 0.0478832
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0632853, upper bound: 0.0664291
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0655431, upper bound: 0.0646759
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0475607, 0.0481719
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0564648, upper bound: 0.0653727
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0647049, upper bound: 0.0573228
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476866, 0.0472689
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0624110, upper bound: 0.0649936
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0635184, upper bound: 0.0646983
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0469416, 0.0477831
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0655260, upper bound: 0.0640283
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0656621, upper bound: 0.0640283
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0462232, 0.0459222
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0640664, upper bound: 0.0688713
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651010, upper bound: 0.0675941
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0456340, 0.0466763
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0658675, upper bound: 0.0653831
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0668178, upper bound: 0.0615253
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0449482, 0.0434618
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0559635, upper bound: 0.0613481
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0566406, upper bound: 0.0557559
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0436201, 0.0448052
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 249

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0653927, upper bound: 0.0579916
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0665452, upper bound: 0.0549125
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477368, 0.0474709
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0633090, upper bound: 0.0660835
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630226, upper bound: 0.0667106
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476421, 0.0477565
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0567613, upper bound: 0.0582156
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0567613, upper bound: 0.0582156
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0470176, 0.0475765
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0662816, upper bound: 0.0601949
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0665537, upper bound: 0.0601475
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477153, 0.0479129
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0552227, upper bound: 0.0673961
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0585742, upper bound: 0.0655899
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0475953, 0.0480417
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0579265, upper bound: 0.0650305
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0593929, upper bound: 0.0597154
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477675, 0.0481738
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0648526, upper bound: 0.0595302
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682709, upper bound: 0.0597989
time: 0.61 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0638442, upper bound: 0.0601145
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0642506, upper bound: 0.0597270
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0606473, upper bound: 0.0638590
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0607891, upper bound: 0.0633788
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0594317, upper bound: 0.0613762
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0640078, upper bound: 0.0586180
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0647085, upper bound: 0.0627792
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0666545, upper bound: 0.0619037
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0563710, upper bound: 0.0587291
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0586971, upper bound: 0.0562493
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0648579, upper bound: 0.0631934
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0665159, upper bound: 0.0629815
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0591080, upper bound: 0.0667014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0625044, upper bound: 0.0579186
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0673543, upper bound: 0.0640645
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0663766, upper bound: 0.0642426
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0654576, upper bound: 0.0626927
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0682986, upper bound: 0.0583959
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0650535, upper bound: 0.0657627
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0650813, upper bound: 0.0666344
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0615007, upper bound: 0.0662932
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0615508, upper bound: 0.0661725
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0676988, upper bound: 0.0640645
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0665282, upper bound: 0.0642662
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0594406, upper bound: 0.0622268
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0671870, upper bound: 0.0572501
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0630192, upper bound: 0.0668411
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0642662, upper bound: 0.0665282
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0650087, upper bound: 0.0645937
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0660170, upper bound: 0.0645937
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0596782, upper bound: 0.0685669
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0646494, upper bound: 0.0613416
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0592594, upper bound: 0.0650960
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0669554, upper bound: 0.0613423
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0648323, upper bound: 0.0652187
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0648495, upper bound: 0.0651141
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0680146, upper bound: 0.0608458
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0659298, upper bound: 0.0659123
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0632853, upper bound: 0.0664291
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0655431, upper bound: 0.0646759
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0564648, upper bound: 0.0653727
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0647049, upper bound: 0.0573228
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0624110, upper bound: 0.0649936
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0635184, upper bound: 0.0646983
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0655260, upper bound: 0.0640283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0656621, upper bound: 0.0640283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0640664, upper bound: 0.0688713
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0651010, upper bound: 0.0675941
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0658675, upper bound: 0.0653831
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0668178, upper bound: 0.0615253
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0559635, upper bound: 0.0613481
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0566406, upper bound: 0.0557559
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0653927, upper bound: 0.0579916
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0665452, upper bound: 0.0549125
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0633090, upper bound: 0.0660835
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0630226, upper bound: 0.0667106
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0567613, upper bound: 0.0582156
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0567613, upper bound: 0.0582156
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0662816, upper bound: 0.0601949
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0665537, upper bound: 0.0601475
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0552227, upper bound: 0.0673961
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0585742, upper bound: 0.0655899
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0579265, upper bound: 0.0650305
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0593929, upper bound: 0.0597154
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0648526, upper bound: 0.0595302
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 7, lower bound: -0.0682709, upper bound: 0.0597989

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0475008, 0.0481462
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0647527, upper bound: 0.0616414
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0664579, upper bound: 0.0615954
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0448684, 0.0435449
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0558983, upper bound: 0.0610261
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0567433, upper bound: 0.0570522
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477273, 0.0477496
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0635837, upper bound: 0.0613665
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0639257, upper bound: 0.0613665
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0455335, 0.0467735
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0654753, upper bound: 0.0554958
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0657282, upper bound: 0.0555111
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476514, 0.0481299
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0638336, upper bound: 0.0663921
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0648234, upper bound: 0.0638370
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476229, 0.0481672
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 249

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0653653, upper bound: 0.0633004
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666893, upper bound: 0.0594739
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0434573, 0.0451989
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0631077, upper bound: 0.0554602
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0635584, upper bound: 0.0554602
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0474903, 0.0472439
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0610525, upper bound: 0.0632173
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0610525, upper bound: 0.0629371
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0449411, 0.0434573
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0560697, upper bound: 0.0652924
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0560744, upper bound: 0.0650866
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0436295, 0.0448029
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0646447, upper bound: 0.0603723
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0658513, upper bound: 0.0571177
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477629, 0.0481726
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640945, upper bound: 0.0580874
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0649971, upper bound: 0.0576465
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477152, 0.0479076
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0595090, upper bound: 0.0663910
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0626158, upper bound: 0.0606361
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0475961, 0.0480393
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 129

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636818, upper bound: 0.0647405
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636856, upper bound: 0.0645319
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0469416, 0.0477831
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Candidate
type: RSZ, layer: 3, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0589604, upper bound: 0.0594788
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0642997, upper bound: 0.0540085
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478299, 0.0476206
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0617474, upper bound: 0.0663323
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0626591, upper bound: 0.0641762
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0476416, 0.0472742
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0551861, upper bound: 0.0648442
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0507296, upper bound: 0.0673629
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0456332, 0.0466762
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0681248, upper bound: 0.0596229
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670691, upper bound: 0.0594637
time: 0.65 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0647527, upper bound: 0.0616414
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0664579, upper bound: 0.0615954
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0558983, upper bound: 0.0610261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0567433, upper bound: 0.0570522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0635837, upper bound: 0.0613665
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0639257, upper bound: 0.0613665
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0654753, upper bound: 0.0554958
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0657282, upper bound: 0.0555111
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0638336, upper bound: 0.0663921
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0648234, upper bound: 0.0638370
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0653653, upper bound: 0.0633004
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0666893, upper bound: 0.0594739
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0631077, upper bound: 0.0554602
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0635584, upper bound: 0.0554602
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0610525, upper bound: 0.0632173
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0610525, upper bound: 0.0629371
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0560697, upper bound: 0.0652924
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0560744, upper bound: 0.0650866
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0646447, upper bound: 0.0603723
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0658513, upper bound: 0.0571177
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0640945, upper bound: 0.0580874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0649971, upper bound: 0.0576465
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0595090, upper bound: 0.0663910
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0626158, upper bound: 0.0606361
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0636818, upper bound: 0.0647405
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0636856, upper bound: 0.0645319
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0589604, upper bound: 0.0594788
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0642997, upper bound: 0.0540085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0617474, upper bound: 0.0663323
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0626591, upper bound: 0.0641762
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0551861, upper bound: 0.0648442
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0507296, upper bound: 0.0673629
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0681248, upper bound: 0.0596229
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 7, lower bound: -0.0670691, upper bound: 0.0594637

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0468487, 0.0479411
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0645848, upper bound: 0.0592196
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0664517, upper bound: 0.0573536
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0478582, 0.0480796
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0490676, upper bound: 0.0662065
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0490692, upper bound: 0.0655318
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477150, 0.0480768
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0647653, upper bound: 0.0562714
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0649057, upper bound: 0.0562183
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0171053, 0.0129703, -0.0171053, 0.0129703, -0.0300755, 0.0300755
1: -0.0310786, 0.0110978, -0.0310786, 0.0110978, -0.0421764, 0.0421764
2: 0.0277538, 0.0660720, 0.0277538, 0.0660720, -0.0383182, 0.0383182
3: 0.0005956, 0.0548688, 0.0005956, 0.0548688, -0.0477505, 0.0480347
4: -0.0243545, 0.0238170, -0.0243545, 0.0238170, -0.0481715, 0.0481715
5: -0.0076450, 0.0397362, -0.0076450, 0.0397362, -0.0473811, 0.0473811
6: -0.0497692, -0.0060545, -0.0497692, -0.0060545, -0.0437148, 0.0437148
7: 0.8601831, 0.9667761, 0.8601831, 0.9667761, -0.1065930, 0.1065930
8: -0.0087638, 0.0439597, -0.0087638, 0.0439597, -0.0527235, 0.0527235
9: -0.0213101, 0.0241830, -0.0213101, 0.0241830, -0.0454932, 0.0454932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0649532, upper bound: 0.0586146
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0659514, upper bound: 0.0528756
time: 0.65 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 4.74 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.74
Output dim: 7, lower bound: -0.0645848, upper bound: 0.0592196
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.74
Output dim: 7, lower bound: -0.0664517, upper bound: 0.0573536
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.74
Output dim: 7, lower bound: -0.0490676, upper bound: 0.0662065
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.74
Output dim: 7, lower bound: -0.0490692, upper bound: 0.0655318
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.74
Output dim: 7, lower bound: -0.0647653, upper bound: 0.0562714
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.74
Output dim: 7, lower bound: -0.0649057, upper bound: 0.0562183
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.74
Output dim: 7, lower bound: -0.0649532, upper bound: 0.0586146
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.74
Output dim: 7, lower bound: -0.0659514, upper bound: 0.0528756

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.01 + 344.01 = 347.01 seconds
