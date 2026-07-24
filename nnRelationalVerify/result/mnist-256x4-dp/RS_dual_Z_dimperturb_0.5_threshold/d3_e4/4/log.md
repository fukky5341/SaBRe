## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00913976


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758)
1: (-0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151)
2: (-0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925)
3: (-0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690)
4: (-0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619)
5: (-0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932)
6: (-0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104)
7: (-0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592)
8: (0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425)
9: (-0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0202706, 0.0202706)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.73 + 2.55 = 4.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0140648, upper bound: 0.0140648

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0137750, upper bound: 0.0138482
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138483, upper bound: 0.0137750
time: 1.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.96
Output dim: 8, lower bound: -0.0137750, upper bound: 0.0138482
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.96
Output dim: 8, lower bound: -0.0138483, upper bound: 0.0137750

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0202706, 0.0202706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135943, upper bound: 0.0136900
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136172, upper bound: 0.0136680
time: 1.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0202706, 0.0202706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136681, upper bound: 0.0136172
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136900, upper bound: 0.0135943
time: 1.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 8, lower bound: -0.0135943, upper bound: 0.0136900
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 8, lower bound: -0.0136172, upper bound: 0.0136680
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 8, lower bound: -0.0136681, upper bound: 0.0136172
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 8, lower bound: -0.0136900, upper bound: 0.0135943

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0201215, 0.0201310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129202, upper bound: 0.0129886
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129202, upper bound: 0.0129886
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0201085, 0.0201436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129228, upper bound: 0.0129854
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129228, upper bound: 0.0129854
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0201436, 0.0201085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129854, upper bound: 0.0129227
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129854, upper bound: 0.0129228
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0201310, 0.0201215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129886, upper bound: 0.0129202
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129886, upper bound: 0.0129203
time: 1.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0129202, upper bound: 0.0129886
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0129202, upper bound: 0.0129886
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0129228, upper bound: 0.0129854
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0129228, upper bound: 0.0129854
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0129854, upper bound: 0.0129227
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0129854, upper bound: 0.0129228
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0129886, upper bound: 0.0129202
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0129886, upper bound: 0.0129203

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0200316, 0.0200111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124722, upper bound: 0.0126570
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125684, upper bound: 0.0125337
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0200016, 0.0201310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124722, upper bound: 0.0126570
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125684, upper bound: 0.0125337
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0200243, 0.0200237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124758, upper bound: 0.0126553
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125726, upper bound: 0.0125310
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0199886, 0.0201436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124758, upper bound: 0.0126553
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125726, upper bound: 0.0125310
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0200752, 0.0199886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125310, upper bound: 0.0125726
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126553, upper bound: 0.0124758
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0200237, 0.0201085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125310, upper bound: 0.0125726
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126553, upper bound: 0.0124758
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0200658, 0.0200016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0125684
time: 2.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126570, upper bound: 0.0124722
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0200111, 0.0201215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0125684
time: 2.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126570, upper bound: 0.0124722
time: 1.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0124722, upper bound: 0.0126570
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0125684, upper bound: 0.0125337
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0124722, upper bound: 0.0126570
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0125684, upper bound: 0.0125337
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0124758, upper bound: 0.0126553
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0125726, upper bound: 0.0125310
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0124758, upper bound: 0.0126553
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0125726, upper bound: 0.0125310
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0125310, upper bound: 0.0125726
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0126553, upper bound: 0.0124758
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0125310, upper bound: 0.0125726
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0126553, upper bound: 0.0124758
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0125684
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0126570, upper bound: 0.0124722
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0125684
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 8, lower bound: -0.0126570, upper bound: 0.0124722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187410, 0.0187071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121581, upper bound: 0.0123673
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121425, upper bound: 0.0123744
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187276, 0.0186485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122826, upper bound: 0.0121997
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122738, upper bound: 0.0122144
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186448, 0.0188253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121581, upper bound: 0.0123673
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121425, upper bound: 0.0123744
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186976, 0.0187666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122826, upper bound: 0.0121997
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122738, upper bound: 0.0122144
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187291, 0.0187197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121585, upper bound: 0.0123673
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121428, upper bound: 0.0123743
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187203, 0.0186570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122835, upper bound: 0.0121997
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122741, upper bound: 0.0122144
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186322, 0.0188378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121585, upper bound: 0.0123673
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121428, upper bound: 0.0123743
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186846, 0.0187751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122835, upper bound: 0.0121997
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122741, upper bound: 0.0122144
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187501, 0.0186846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122144, upper bound: 0.0122741
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121997, upper bound: 0.0122835
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187712, 0.0186322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123743, upper bound: 0.0121428
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123673, upper bound: 0.0121585
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186570, 0.0188027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122144, upper bound: 0.0122741
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121997, upper bound: 0.0122836
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187197, 0.0187503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123743, upper bound: 0.0121428
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123673, upper bound: 0.0121585
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187413, 0.0186976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122144, upper bound: 0.0122738
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121997, upper bound: 0.0122826
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187618, 0.0186448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123744, upper bound: 0.0121425
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123673, upper bound: 0.0121582
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186485, 0.0188158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122144, upper bound: 0.0122738
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121997, upper bound: 0.0122826
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187071, 0.0187629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123744, upper bound: 0.0121425
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123673, upper bound: 0.0121582
time: 1.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121581, upper bound: 0.0123673
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121425, upper bound: 0.0123744
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122826, upper bound: 0.0121997
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122738, upper bound: 0.0122144
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121581, upper bound: 0.0123673
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121425, upper bound: 0.0123744
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122826, upper bound: 0.0121997
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122738, upper bound: 0.0122144
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121585, upper bound: 0.0123673
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121428, upper bound: 0.0123743
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122835, upper bound: 0.0121997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122741, upper bound: 0.0122144
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121585, upper bound: 0.0123673
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121428, upper bound: 0.0123743
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122835, upper bound: 0.0121997
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122741, upper bound: 0.0122144
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122144, upper bound: 0.0122741
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121997, upper bound: 0.0122835
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0123743, upper bound: 0.0121428
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0123673, upper bound: 0.0121585
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122144, upper bound: 0.0122741
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121997, upper bound: 0.0122836
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0123743, upper bound: 0.0121428
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0123673, upper bound: 0.0121585
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122144, upper bound: 0.0122738
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121997, upper bound: 0.0122826
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0123744, upper bound: 0.0121425
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0123673, upper bound: 0.0121582
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0122144, upper bound: 0.0122738
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0121997, upper bound: 0.0122826
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0123744, upper bound: 0.0121425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 8, lower bound: -0.0123673, upper bound: 0.0121582

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187304, 0.0186984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0122824
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120499, upper bound: 0.0122048
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187322, 0.0186967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0122898
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120329, upper bound: 0.0122093
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187169, 0.0186397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121541, upper bound: 0.0120997
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121870, upper bound: 0.0120802
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187189, 0.0186379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121467, upper bound: 0.0121137
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121789, upper bound: 0.0120956
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186344, 0.0188165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0122824
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120499, upper bound: 0.0122048
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186360, 0.0188148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0122898
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120329, upper bound: 0.0122093
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186865, 0.0187578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121541, upper bound: 0.0120997
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121870, upper bound: 0.0120802
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186889, 0.0187560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121467, upper bound: 0.0121137
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121789, upper bound: 0.0120955
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187190, 0.0187110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0122824
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120512, upper bound: 0.0122048
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187203, 0.0187092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0122897
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120335, upper bound: 0.0122093
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187099, 0.0186483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121542, upper bound: 0.0120997
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121879, upper bound: 0.0120802
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187116, 0.0186470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121468, upper bound: 0.0121136
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121796, upper bound: 0.0120951
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186219, 0.0188290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0122824
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120512, upper bound: 0.0122048
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186234, 0.0188273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0122897
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120335, upper bound: 0.0122093
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186736, 0.0187664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121542, upper bound: 0.0120997
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121879, upper bound: 0.0120802
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186759, 0.0187650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121468, upper bound: 0.0121137
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121796, upper bound: 0.0120951
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187396, 0.0186759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120951, upper bound: 0.0121796
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121137, upper bound: 0.0121468
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187414, 0.0186736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120802, upper bound: 0.0121879
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120997, upper bound: 0.0121542
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187608, 0.0186234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122093, upper bound: 0.0120335
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122897, upper bound: 0.0120302
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187624, 0.0186219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122048, upper bound: 0.0120512
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122824, upper bound: 0.0120452
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186470, 0.0187939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120951, upper bound: 0.0121796
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121137, upper bound: 0.0121468
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186483, 0.0187917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120802, upper bound: 0.0121879
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120997, upper bound: 0.0121542
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187092, 0.0187415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122093, upper bound: 0.0120335
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122897, upper bound: 0.0120302
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187110, 0.0187400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122048, upper bound: 0.0120512
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122824, upper bound: 0.0120452
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187308, 0.0186889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120956, upper bound: 0.0121789
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121137, upper bound: 0.0121467
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187326, 0.0186865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120802, upper bound: 0.0121870
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120997, upper bound: 0.0121541
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187512, 0.0186360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122093, upper bound: 0.0120329
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122898, upper bound: 0.0120299
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0187531, 0.0186344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122048, upper bound: 0.0120499
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122824, upper bound: 0.0120452
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186379, 0.0188070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120956, upper bound: 0.0121789
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121137, upper bound: 0.0121467
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186397, 0.0188046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120802, upper bound: 0.0121870
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120997, upper bound: 0.0121541
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186967, 0.0187541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122093, upper bound: 0.0120329
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122898, upper bound: 0.0120299
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0186984, 0.0187525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122048, upper bound: 0.0120499
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122824, upper bound: 0.0120452
time: 1.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0122824
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120499, upper bound: 0.0122048
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0122898
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120329, upper bound: 0.0122093
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121541, upper bound: 0.0120997
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121870, upper bound: 0.0120802
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121467, upper bound: 0.0121137
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121789, upper bound: 0.0120956
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0122824
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120499, upper bound: 0.0122048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0122898
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120329, upper bound: 0.0122093
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121541, upper bound: 0.0120997
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121870, upper bound: 0.0120802
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121467, upper bound: 0.0121137
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121789, upper bound: 0.0120955
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0122824
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120512, upper bound: 0.0122048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0122897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120335, upper bound: 0.0122093
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121542, upper bound: 0.0120997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121879, upper bound: 0.0120802
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121468, upper bound: 0.0121136
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121796, upper bound: 0.0120951
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0122824
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120512, upper bound: 0.0122048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0122897
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120335, upper bound: 0.0122093
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121542, upper bound: 0.0120997
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121879, upper bound: 0.0120802
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121468, upper bound: 0.0121137
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121796, upper bound: 0.0120951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120951, upper bound: 0.0121796
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121137, upper bound: 0.0121468
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120802, upper bound: 0.0121879
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120997, upper bound: 0.0121542
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122093, upper bound: 0.0120335
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122897, upper bound: 0.0120302
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122048, upper bound: 0.0120512
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122824, upper bound: 0.0120452
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120951, upper bound: 0.0121796
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121137, upper bound: 0.0121468
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120802, upper bound: 0.0121879
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120997, upper bound: 0.0121542
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122093, upper bound: 0.0120335
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122897, upper bound: 0.0120302
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122048, upper bound: 0.0120512
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122824, upper bound: 0.0120452
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120956, upper bound: 0.0121789
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121137, upper bound: 0.0121467
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120802, upper bound: 0.0121870
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120997, upper bound: 0.0121541
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122093, upper bound: 0.0120329
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122898, upper bound: 0.0120299
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122048, upper bound: 0.0120499
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122824, upper bound: 0.0120452
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120956, upper bound: 0.0121789
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0121137, upper bound: 0.0121467
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120802, upper bound: 0.0121870
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0120997, upper bound: 0.0121541
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122093, upper bound: 0.0120329
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122898, upper bound: 0.0120299
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122048, upper bound: 0.0120499
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.93
Output dim: 8, lower bound: -0.0122824, upper bound: 0.0120452

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184655, 0.0183916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115800, upper bound: 0.0120024
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117595, upper bound: 0.0117387
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184237, 0.0183651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115800, upper bound: 0.0119162
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117646, upper bound: 0.0116975
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184673, 0.0183899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115516, upper bound: 0.0120091
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117422, upper bound: 0.0117534
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184255, 0.0183635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115516, upper bound: 0.0119218
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117476, upper bound: 0.0117113
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184016, 0.0183330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116714, upper bound: 0.0118184
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118682, upper bound: 0.0115931
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184101, 0.0183453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116957, upper bound: 0.0117913
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119015, upper bound: 0.0115822
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184036, 0.0183311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116571, upper bound: 0.0118346
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118620, upper bound: 0.0116255
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184121, 0.0183433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116856, upper bound: 0.0118085
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118926, upper bound: 0.0116175
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183588, 0.0185086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115800, upper bound: 0.0120024
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117595, upper bound: 0.0117387
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183276, 0.0184820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115800, upper bound: 0.0119162
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117646, upper bound: 0.0116975
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183605, 0.0185069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115516, upper bound: 0.0120091
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117422, upper bound: 0.0117534
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183293, 0.0184805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115516, upper bound: 0.0119218
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117476, upper bound: 0.0117113
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183649, 0.0184499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116714, upper bound: 0.0118184
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118682, upper bound: 0.0115931
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183797, 0.0184623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116957, upper bound: 0.0117913
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119015, upper bound: 0.0115822
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183666, 0.0184481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116571, upper bound: 0.0118346
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118620, upper bound: 0.0116255
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183821, 0.0184603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116856, upper bound: 0.0118085
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118926, upper bound: 0.0116175
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184538, 0.0184042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115800, upper bound: 0.0120024
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117604, upper bound: 0.0117387
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184123, 0.0183796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115801, upper bound: 0.0119160
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117651, upper bound: 0.0116975
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184556, 0.0184025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115516, upper bound: 0.0120091
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117436, upper bound: 0.0117534
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184136, 0.0183777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115516, upper bound: 0.0119216
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117485, upper bound: 0.0117113
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183922, 0.0183415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116714, upper bound: 0.0118165
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118697, upper bound: 0.0115931
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184031, 0.0183599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116957, upper bound: 0.0117908
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0115822
time: 2.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183939, 0.0183402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116571, upper bound: 0.0118332
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118633, upper bound: 0.0116255
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184048, 0.0183581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116856, upper bound: 0.0118075
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118945, upper bound: 0.0116175
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183478, 0.0185211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115800, upper bound: 0.0120024
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117604, upper bound: 0.0117387
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183152, 0.0184965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115801, upper bound: 0.0119160
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117651, upper bound: 0.0116975
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183498, 0.0185194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115516, upper bound: 0.0120091
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117436, upper bound: 0.0117534
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183167, 0.0184947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115516, upper bound: 0.0119216
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117485, upper bound: 0.0117113
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183590, 0.0184584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116714, upper bound: 0.0118165
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118697, upper bound: 0.0115931
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183668, 0.0184768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116957, upper bound: 0.0117908
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0115822
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183609, 0.0184571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116571, upper bound: 0.0118332
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118633, upper bound: 0.0116255
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183691, 0.0184751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116856, upper bound: 0.0118075
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118945, upper bound: 0.0116175
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184584, 0.0183691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116175, upper bound: 0.0118945
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118075, upper bound: 0.0116856
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184328, 0.0183609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116255, upper bound: 0.0118633
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118332, upper bound: 0.0116571
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184597, 0.0183668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115822, upper bound: 0.0119030
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117908, upper bound: 0.0116957
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184346, 0.0183590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115931, upper bound: 0.0118697
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118165, upper bound: 0.0116714
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184331, 0.0183167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117113, upper bound: 0.0117484
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119216, upper bound: 0.0115516
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184540, 0.0183498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117534, upper bound: 0.0117436
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120091, upper bound: 0.0115516
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184347, 0.0183152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116975, upper bound: 0.0117651
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119160, upper bound: 0.0115801
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0184557, 0.0183478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117387, upper bound: 0.0117604
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120024, upper bound: 0.0115800
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183581, 0.0184860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116175, upper bound: 0.0118945
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118075, upper bound: 0.0116856
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183402, 0.0184778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116255, upper bound: 0.0118633
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118332, upper bound: 0.0116571
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183599, 0.0184837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115822, upper bound: 0.0119030
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117908, upper bound: 0.0116957
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023994, 0.0241752, 0.0023994, 0.0241752, -0.0217758, 0.0217758
1: -0.0055582, 0.0043569, -0.0055582, 0.0043569, -0.0099151, 0.0099151
2: -0.0019592, 0.0130333, -0.0019592, 0.0130333, -0.0149925, 0.0149925
3: -0.0074682, 0.0049008, -0.0074682, 0.0049008, -0.0123690, 0.0123690
4: -0.0036742, 0.0022876, -0.0036742, 0.0022876, -0.0059619, 0.0059619
5: -0.0034897, 0.0067035, -0.0034897, 0.0067035, -0.0101932, 0.0101932
6: -0.0189134, 0.0042970, -0.0189134, 0.0042970, -0.0232104, 0.0232104
7: -0.0159311, 0.0189281, -0.0159311, 0.0189281, -0.0348592, 0.0348592
8: 0.9800811, 1.0034236, 0.9800811, 1.0034236, -0.0233425, 0.0233425
9: -0.0171491, 0.0031215, -0.0171491, 0.0031215, -0.0183415, 0.0184760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.28 + 597.64 = 601.92 seconds
