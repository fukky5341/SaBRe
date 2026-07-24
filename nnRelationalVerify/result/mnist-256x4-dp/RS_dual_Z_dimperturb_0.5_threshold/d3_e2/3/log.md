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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745)
1: (-0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206)
2: (-0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471)
3: (-0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4641285, 0.4641284)
4: (-0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983)
5: (-0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583)
6: (-0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172)
7: (0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302)
8: (-0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574)
9: (-0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.51 + 2.19 = 3.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4833092, upper bound: 0.4833092

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4811742, upper bound: 0.4812279
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4812279, upper bound: 0.4811742
time: 1.01 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 7, lower bound: -0.4811742, upper bound: 0.4812279
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 7, lower bound: -0.4812279, upper bound: 0.4811742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4638930, 0.4638229
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4275041, upper bound: 0.4276143
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4275041, upper bound: 0.4276143
time: 0.90 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4638229, 0.4638930
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4276143, upper bound: 0.4275041
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4276143, upper bound: 0.4275041
time: 1.13 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 7, lower bound: -0.4275041, upper bound: 0.4276143
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 7, lower bound: -0.4275041, upper bound: 0.4276143
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 7, lower bound: -0.4276143, upper bound: 0.4275041
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 7, lower bound: -0.4276143, upper bound: 0.4275041

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636306, 0.4636499
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858621, upper bound: 0.3858083
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858621, upper bound: 0.3858083
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637200, 0.4638229
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858621, upper bound: 0.3858083
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858621, upper bound: 0.3858083
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635609, 0.4637199
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858083, upper bound: 0.3858621
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858083, upper bound: 0.3858621
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636499, 0.4638930
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858083, upper bound: 0.3858621
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858083, upper bound: 0.3858621
time: 0.70 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 7, lower bound: -0.3858621, upper bound: 0.3858083
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 7, lower bound: -0.3858621, upper bound: 0.3858083
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 7, lower bound: -0.3858621, upper bound: 0.3858083
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 7, lower bound: -0.3858621, upper bound: 0.3858083
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 7, lower bound: -0.3858083, upper bound: 0.3858621
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 7, lower bound: -0.3858083, upper bound: 0.3858621
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 7, lower bound: -0.3858083, upper bound: 0.3858621
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 7, lower bound: -0.3858083, upper bound: 0.3858621

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635736, 0.4637165
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3857153, upper bound: 0.3842780
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3844033, upper bound: 0.3856617
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636306, 0.4635929
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3857153, upper bound: 0.3842780
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3844033, upper bound: 0.3856617
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636630, 0.4638895
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3857153, upper bound: 0.3842780
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3844033, upper bound: 0.3856617
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637200, 0.4637659
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3857153, upper bound: 0.3842780
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3844033, upper bound: 0.3856617
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635040, 0.4637890
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3856617, upper bound: 0.3844033
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3842780, upper bound: 0.3857153
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635609, 0.4636629
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3856617, upper bound: 0.3844033
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3842780, upper bound: 0.3857153
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635929, 0.4639620
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3856617, upper bound: 0.3844033
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3842780, upper bound: 0.3857153
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636499, 0.4638360
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3856617, upper bound: 0.3844033
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3842780, upper bound: 0.3857153
time: 0.83 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3857153, upper bound: 0.3842780
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3844033, upper bound: 0.3856617
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3857153, upper bound: 0.3842780
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3844033, upper bound: 0.3856617
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3857153, upper bound: 0.3842780
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3844033, upper bound: 0.3856617
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3857153, upper bound: 0.3842780
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3844033, upper bound: 0.3856617
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3856617, upper bound: 0.3844033
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3842780, upper bound: 0.3857153
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3856617, upper bound: 0.3844033
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3842780, upper bound: 0.3857153
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3856617, upper bound: 0.3844033
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3842780, upper bound: 0.3857153
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3856617, upper bound: 0.3844033
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 7, lower bound: -0.3842780, upper bound: 0.3857153

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636290, 0.4638197
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3844374, upper bound: 0.3822539
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3836371, upper bound: 0.3830111
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636768, 0.4637682
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3831217, upper bound: 0.3836115
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3823143, upper bound: 0.3843779
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636850, 0.4636961
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3844374, upper bound: 0.3822539
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3836371, upper bound: 0.3830111
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637328, 0.4636485
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3831217, upper bound: 0.3836115
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3823143, upper bound: 0.3843779
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637175, 0.4639927
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3844374, upper bound: 0.3822539
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3836371, upper bound: 0.3830111
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637661, 0.4639412
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3831217, upper bound: 0.3836115
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3823143, upper bound: 0.3843779
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637735, 0.4638690
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3844374, upper bound: 0.3822539
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3836371, upper bound: 0.3830111
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4638221, 0.4638214
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3831217, upper bound: 0.3836115
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3823143, upper bound: 0.3843779
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635557, 0.4638922
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3843779, upper bound: 0.3823143
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3836115, upper bound: 0.3831217
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636072, 0.4638416
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3830111, upper bound: 0.3836371
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3822539, upper bound: 0.3844374
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636117, 0.4637662
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3843779, upper bound: 0.3823143
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3836115, upper bound: 0.3831217
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636632, 0.4637175
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3830111, upper bound: 0.3836371
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3822539, upper bound: 0.3844374
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636486, 0.4640651
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3843779, upper bound: 0.3823143
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3836115, upper bound: 0.3831217
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636961, 0.4640145
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3830111, upper bound: 0.3836371
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3822539, upper bound: 0.3844374
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637045, 0.4639391
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3843779, upper bound: 0.3823143
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3836115, upper bound: 0.3831217
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637522, 0.4638904
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3830111, upper bound: 0.3836371
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3822539, upper bound: 0.3844374
time: 0.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3844374, upper bound: 0.3822539
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3836371, upper bound: 0.3830111
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3831217, upper bound: 0.3836115
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3823143, upper bound: 0.3843779
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3844374, upper bound: 0.3822539
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3836371, upper bound: 0.3830111
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3831217, upper bound: 0.3836115
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3823143, upper bound: 0.3843779
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3844374, upper bound: 0.3822539
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3836371, upper bound: 0.3830111
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3831217, upper bound: 0.3836115
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3823143, upper bound: 0.3843779
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3844374, upper bound: 0.3822539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3836371, upper bound: 0.3830111
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3831217, upper bound: 0.3836115
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3823143, upper bound: 0.3843779
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3843779, upper bound: 0.3823143
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3836115, upper bound: 0.3831217
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3830111, upper bound: 0.3836371
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3822539, upper bound: 0.3844374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3843779, upper bound: 0.3823143
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3836115, upper bound: 0.3831217
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3830111, upper bound: 0.3836371
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3822539, upper bound: 0.3844374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3843779, upper bound: 0.3823143
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3836115, upper bound: 0.3831217
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3830111, upper bound: 0.3836371
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3822539, upper bound: 0.3844374
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3843779, upper bound: 0.3823143
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3836115, upper bound: 0.3831217
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3830111, upper bound: 0.3836371
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 7, lower bound: -0.3822539, upper bound: 0.3844374

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635564, 0.4637655
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635735, 0.4637471
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636042, 0.4637134
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636208, 0.4636956
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636124, 0.4636444
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636294, 0.4636235
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636601, 0.4635958
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636766, 0.4635758
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636449, 0.4639386
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636649, 0.4639202
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636935, 0.4638865
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637135, 0.4638687
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637009, 0.4638175
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637209, 0.4637966
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637495, 0.4637689
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637693, 0.4637489
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4634831, 0.4638395
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635011, 0.4638196
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635346, 0.4637883
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635549, 0.4637689
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635390, 0.4637135
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635569, 0.4636935
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635905, 0.4636649
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636108, 0.4636449
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635758, 0.4640126
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4635959, 0.4639927
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636235, 0.4639614
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636444, 0.4639420
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636317, 0.4638866
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636518, 0.4638666
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4636794, 0.4638380
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4637004, 0.4638180
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
time: 0.79 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3756592
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762779
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3769725
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3777363
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3777363, upper bound: 0.3757378
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3769725, upper bound: 0.3764378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3762779, upper bound: 0.3770277
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.12
Output dim: 7, lower bound: -0.3756592, upper bound: 0.3778318

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4631833, 0.4634951
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3755695
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775678, upper bound: 0.3756592
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4632860, 0.4637655
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3755695
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775678, upper bound: 0.3756592
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4632000, 0.4634767
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762211
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3768191, upper bound: 0.3762779
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633031, 0.4637471
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762211
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3768191, upper bound: 0.3762779
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4632328, 0.4634430
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3767325
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764060, upper bound: 0.3769725
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633337, 0.4637134
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3767325
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764060, upper bound: 0.3769725
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4632484, 0.4634252
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3774262
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756529, upper bound: 0.3777363
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633503, 0.4636956
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3774262
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756529, upper bound: 0.3777363
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4632392, 0.4633740
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3755695
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775678, upper bound: 0.3756592
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633420, 0.4636444
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3755695
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775678, upper bound: 0.3756592
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4632559, 0.4633530
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762211
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3768191, upper bound: 0.3762779
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633589, 0.4636235
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762211
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3768191, upper bound: 0.3762779
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4632888, 0.4633254
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3767325
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764060, upper bound: 0.3769725
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633896, 0.4635958
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3767325
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764060, upper bound: 0.3769725
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633043, 0.4633054
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3774262
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756529, upper bound: 0.3777363
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4634063, 0.4635758
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3774262
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756529, upper bound: 0.3777363
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4632756, 0.4636686
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3755695
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775678, upper bound: 0.3756592
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633745, 0.4639386
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3755695
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775678, upper bound: 0.3756592
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4632957, 0.4636502
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762211
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3768191, upper bound: 0.3762779
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633945, 0.4639202
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762211
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3768191, upper bound: 0.3762779
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633191, 0.4636165
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3767325
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764060, upper bound: 0.3769725
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4634231, 0.4638865
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764378, upper bound: 0.3767325
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3764060, upper bound: 0.3769725
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633394, 0.4635987
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3774262
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756529, upper bound: 0.3777363
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4634430, 0.4638687
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3757378, upper bound: 0.3774262
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756529, upper bound: 0.3777363
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633315, 0.4635475
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3755695
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775678, upper bound: 0.3756592
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4634304, 0.4638175
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3778318, upper bound: 0.3755695
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775678, upper bound: 0.3756592
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4633517, 0.4635266
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770277, upper bound: 0.3762211
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3768191, upper bound: 0.3762779
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2275861, 0.2070884, -0.2275861, 0.2070884, -0.4346745, 0.4346745
1: -0.2093681, 0.1973525, -0.2093681, 0.1973525, -0.4067206, 0.4067206
2: -0.1505241, 0.3091230, -0.1505241, 0.3091230, -0.4596471, 0.4596471
3: -0.1270398, 0.3425112, -0.1270398, 0.3425112, -0.4634504, 0.4637966
4: -0.2265072, 0.2282912, -0.2265072, 0.2282912, -0.4547983, 0.4547983
5: -0.2399215, 0.2577368, -0.2399215, 0.2577368, -0.4976583, 0.4976583
6: -0.2180807, 0.2370365, -0.2180807, 0.2370365, -0.4551172, 0.4551172
7: 0.4926099, 1.0818400, 0.4926099, 1.0818400, -0.5892302, 0.5892302
8: -0.1680741, 0.2970833, -0.1680741, 0.2970833, -0.4651574, 0.4651574
9: -0.2238385, 0.2857978, -0.2238385, 0.2857978, -0.5096363, 0.5096363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.69 + 597.50 = 601.19 seconds
