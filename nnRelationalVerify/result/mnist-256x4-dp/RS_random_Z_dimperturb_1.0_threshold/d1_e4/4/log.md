## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.45381564


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095)
1: (-0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130)
2: (-0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546)
3: (-0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4351088, 0.4351088)
4: (-0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590)
5: (-0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786)
6: (-0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493)
7: (0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275)
8: (-0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3312583, 0.3312583)
9: (-0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.24 + 2.56 = 3.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.5042396, upper bound: 0.5042396

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5039671, upper bound: 0.5023511
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5023511, upper bound: 0.5039671
time: 1.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.27
Output dim: 7, lower bound: -0.5039671, upper bound: 0.5023511
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.27
Output dim: 7, lower bound: -0.5023511, upper bound: 0.5039671

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4347478, 0.4348446
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3304650, 0.3306428
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5016188, upper bound: 0.5000848
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5017582, upper bound: 0.4993732
time: 2.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4348446, 0.4347479
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3306428, 0.3304650
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4993556, upper bound: 0.5016183
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5000215, upper bound: 0.5009853
time: 1.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 9.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 9.70
Output dim: 7, lower bound: -0.5016188, upper bound: 0.5000848
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 9.70
Output dim: 7, lower bound: -0.5017582, upper bound: 0.4993732
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 9.70
Output dim: 7, lower bound: -0.4993556, upper bound: 0.5016183
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 9.70
Output dim: 7, lower bound: -0.5000215, upper bound: 0.5009853

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4345789, 0.4346808
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3301328, 0.3303204
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4992045, upper bound: 0.4976467
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4991667, upper bound: 0.4977037
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4345841, 0.4346709
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3301426, 0.3303021
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4891689, upper bound: 0.4870566
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4890073, upper bound: 0.4871417
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4345591, 0.4344341
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3300048, 0.3297746
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4752376, upper bound: 0.4766702
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4752376, upper bound: 0.4766702
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4345307, 0.4344599
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3299524, 0.3298223
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4550717, upper bound: 0.4557715
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4550717, upper bound: 0.4557715
time: 1.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 7, lower bound: -0.4992045, upper bound: 0.4976467
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 7, lower bound: -0.4991667, upper bound: 0.4977037
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 7, lower bound: -0.4891689, upper bound: 0.4870566
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 7, lower bound: -0.4890073, upper bound: 0.4871417
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 7, lower bound: -0.4752376, upper bound: 0.4766702
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 7, lower bound: -0.4752376, upper bound: 0.4766702
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 7, lower bound: -0.4550717, upper bound: 0.4557715
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 7, lower bound: -0.4550717, upper bound: 0.4557715

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4344158, 0.4345344
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3298515, 0.3300698
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4983517, upper bound: 0.4967019
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4982573, upper bound: 0.4967652
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4344325, 0.4345170
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3298823, 0.3300377
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4938421, upper bound: 0.4926845
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4941616, upper bound: 0.4923813
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343061, 0.4344694
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3296439, 0.3299444
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4443203, upper bound: 0.4436294
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4443203, upper bound: 0.4436294
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343822, 0.4343929
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3297840, 0.3298036
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4443203, upper bound: 0.4436294
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4443203, upper bound: 0.4436294
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4345362, 0.4342060
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3299636, 0.3293559
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4437978, upper bound: 0.4445033
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4437978, upper bound: 0.4445033
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343312, 0.4344341
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295861, 0.3297746
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4725435, upper bound: 0.4742256
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728490, upper bound: 0.4741634
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343306, 0.4344293
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295913, 0.3297729
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4263477, upper bound: 0.4267179
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4263477, upper bound: 0.4267179
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4345307, 0.4342599
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3299524, 0.3294612
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4540291, upper bound: 0.4530718
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4527853, upper bound: 0.4547696
time: 1.19 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4983517, upper bound: 0.4967019
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4982573, upper bound: 0.4967652
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4938421, upper bound: 0.4926845
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4941616, upper bound: 0.4923813
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4443203, upper bound: 0.4436294
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4443203, upper bound: 0.4436294
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4443203, upper bound: 0.4436294
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4443203, upper bound: 0.4436294
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4437978, upper bound: 0.4445033
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4437978, upper bound: 0.4445033
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4725435, upper bound: 0.4742256
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4728490, upper bound: 0.4741634
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4263477, upper bound: 0.4267179
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4263477, upper bound: 0.4267179
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4540291, upper bound: 0.4530718
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 7, lower bound: -0.4527853, upper bound: 0.4547696

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343942, 0.4345157
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3298177, 0.3300412
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4700882, upper bound: 0.4691510
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4700882, upper bound: 0.4691510
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343959, 0.4345128
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3298207, 0.3300359
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4952025, upper bound: 0.4945531
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4959872, upper bound: 0.4936693
time: 2.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342710, 0.4343292
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3296009, 0.3297080
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4932462, upper bound: 0.4919966
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4930789, upper bound: 0.4920858
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342448, 0.4343520
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295525, 0.3297499
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4914561, upper bound: 0.4902798
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4921460, upper bound: 0.4898061
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341571, 0.4342706
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3292319, 0.3294389
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4412650, upper bound: 0.4420906
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4412650, upper bound: 0.4420906
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341676, 0.4342653
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3292511, 0.3294292
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702441, upper bound: 0.4712976
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4699793, upper bound: 0.4715788
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4328036, 0.4328216
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7632408, 0.7623888
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3265186, 0.3265623
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4340069, upper bound: 0.4332475
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4340069, upper bound: 0.4332475
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4330862, 0.4325327
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7584937, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3270385, 0.3260307
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4328326, upper bound: 0.4347971
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4328326, upper bound: 0.4347971
time: 1.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4700882, upper bound: 0.4691510
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4700882, upper bound: 0.4691510
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4952025, upper bound: 0.4945531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4959872, upper bound: 0.4936693
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4932462, upper bound: 0.4919966
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4930789, upper bound: 0.4920858
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4914561, upper bound: 0.4902798
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4921460, upper bound: 0.4898061
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4412650, upper bound: 0.4420906
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4412650, upper bound: 0.4420906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4702441, upper bound: 0.4712976
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4699793, upper bound: 0.4715788
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4340069, upper bound: 0.4332475
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4340069, upper bound: 0.4332475
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4328326, upper bound: 0.4347971
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.4328326, upper bound: 0.4347971

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343733, 0.4345029
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3297806, 0.3300192
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4634686, upper bound: 0.4624921
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4634923, upper bound: 0.4624184
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343814, 0.4344947
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3297957, 0.3300042
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4309003, upper bound: 0.4303360
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4309003, upper bound: 0.4303360
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341246, 0.4342090
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3292082, 0.3293634
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4502586, upper bound: 0.4497504
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4502586, upper bound: 0.4497504
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4340920, 0.4342375
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3291482, 0.3294159
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4058442, upper bound: 0.4051189
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4058442, upper bound: 0.4051189
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342493, 0.4343095
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295645, 0.3296751
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4463737, upper bound: 0.4454763
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4463737, upper bound: 0.4454763
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342513, 0.4343076
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295682, 0.3296716
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4679219, upper bound: 0.4679585
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4679219, upper bound: 0.4679585
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4339688, 0.4340495
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3289288, 0.3290776
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3994107, upper bound: 0.3986544
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3994107, upper bound: 0.3986544
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4339423, 0.4340852
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3288801, 0.3291432
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4613708, upper bound: 0.4601406
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4613708, upper bound: 0.4601406
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4340210, 0.4341301
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3290122, 0.3292111
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4682676, upper bound: 0.4664434
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4659409, upper bound: 0.4693440
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4340323, 0.4341134
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3290330, 0.3291804
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4653796, upper bound: 0.4676105
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4658832, upper bound: 0.4672969
time: 1.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4634686, upper bound: 0.4624921
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4634923, upper bound: 0.4624184
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4309003, upper bound: 0.4303360
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4309003, upper bound: 0.4303360
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4502586, upper bound: 0.4497504
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4502586, upper bound: 0.4497504
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4058442, upper bound: 0.4051189
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4058442, upper bound: 0.4051189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4463737, upper bound: 0.4454763
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4463737, upper bound: 0.4454763
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4679219, upper bound: 0.4679585
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4679219, upper bound: 0.4679585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.3994107, upper bound: 0.3986544
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.3994107, upper bound: 0.3986544
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4613708, upper bound: 0.4601406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4613708, upper bound: 0.4601406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4682676, upper bound: 0.4664434
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4659409, upper bound: 0.4693440
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4653796, upper bound: 0.4676105
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.50
Output dim: 7, lower bound: -0.4658832, upper bound: 0.4672969

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342119, 0.4343150
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3294972, 0.3296871
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4257023, upper bound: 0.4254921
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4257023, upper bound: 0.4254921
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341854, 0.4343382
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3294485, 0.3297297
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4261305, upper bound: 0.4249606
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4261305, upper bound: 0.4249606
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342091, 0.4340794
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3294897, 0.3292511
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4253908, upper bound: 0.4255545
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4253908, upper bound: 0.4255545
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4340231, 0.4343076
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3291477, 0.3296716
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4253908, upper bound: 0.4255545
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4253908, upper bound: 0.4255545
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4339211, 0.4340717
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3288422, 0.3291191
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4611192, upper bound: 0.4598641
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4611200, upper bound: 0.4598937
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4339294, 0.4340641
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3288572, 0.3291052
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4387219, upper bound: 0.4373498
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4387219, upper bound: 0.4373498
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4322867, 0.4326794
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7565365, 0.7501171
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3255639, 0.3262824
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4382926, upper bound: 0.4377540
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4382926, upper bound: 0.4377540
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4325764, 0.4323961
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7518800, 0.7548771
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3260968, 0.3257611
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4614584, upper bound: 0.4653959
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619046, upper bound: 0.4649214
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4338678, 0.4339256
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3287455, 0.3288495
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4651028, upper bound: 0.4668634
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4649016, upper bound: 0.4671116
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4338444, 0.4339522
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3287024, 0.3288983
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4234075, upper bound: 0.4238629
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4234075, upper bound: 0.4238629
time: 0.89 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4257023, upper bound: 0.4254921
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4257023, upper bound: 0.4254921
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4261305, upper bound: 0.4249606
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4261305, upper bound: 0.4249606
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4253908, upper bound: 0.4255545
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4253908, upper bound: 0.4255545
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4253908, upper bound: 0.4255545
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4253908, upper bound: 0.4255545
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4611192, upper bound: 0.4598641
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4611200, upper bound: 0.4598937
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4387219, upper bound: 0.4373498
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4387219, upper bound: 0.4373498
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4382926, upper bound: 0.4377540
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4382926, upper bound: 0.4377540
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4614584, upper bound: 0.4653959
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4619046, upper bound: 0.4649214
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4651028, upper bound: 0.4668634
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4649016, upper bound: 0.4671116
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4234075, upper bound: 0.4238629
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.92
Output dim: 7, lower bound: -0.4234075, upper bound: 0.4238629

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4338982, 0.4340497
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3288050, 0.3290837
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4384781, upper bound: 0.4372716
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4384781, upper bound: 0.4372716
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4339005, 0.4340487
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3288091, 0.3290819
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4252818, upper bound: 0.4240589
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4252818, upper bound: 0.4240589
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4324110, 0.4322079
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7486636, 0.7520522
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3257991, 0.3254226
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4610611, upper bound: 0.4647246
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4606593, upper bound: 0.4648956
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4323883, 0.4322341
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7490954, 0.7516794
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3257574, 0.3254710
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4285972, upper bound: 0.4298877
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4285972, upper bound: 0.4298877
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4338448, 0.4339043
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3287081, 0.3288155
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4631399, upper bound: 0.4620251
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4605695, upper bound: 0.4649571
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4338468, 0.4339027
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3287116, 0.3288125
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4629413, upper bound: 0.4622409
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4602307, upper bound: 0.4651932
time: 1.11 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4384781, upper bound: 0.4372716
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4384781, upper bound: 0.4372716
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4252818, upper bound: 0.4240589
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4252818, upper bound: 0.4240589
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4610611, upper bound: 0.4647246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4606593, upper bound: 0.4648956
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4285972, upper bound: 0.4298877
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4285972, upper bound: 0.4298877
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4631399, upper bound: 0.4620251
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4605695, upper bound: 0.4649571
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4629413, upper bound: 0.4622409
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 7, lower bound: -0.4602307, upper bound: 0.4651932

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4323882, 0.4321875
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7479941, 0.7513455
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3257629, 0.3253907
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4209202, upper bound: 0.4228556
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4209202, upper bound: 0.4228556
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4323887, 0.4321852
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7479575, 0.7513524
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3257636, 0.3253866
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4285075, upper bound: 0.4298566
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4285075, upper bound: 0.4298566
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4321105, 0.4324524
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7523481, 0.7467817
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3252518, 0.3258782
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4214865, upper bound: 0.4218887
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4214865, upper bound: 0.4218887
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4323951, 0.4321702
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7477108, 0.7514585
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3257755, 0.3253590
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4283330, upper bound: 0.4300892
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4283330, upper bound: 0.4300892
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4321125, 0.4324513
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7523290, 0.7468135
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3252554, 0.3258761
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4289899, upper bound: 0.4290068
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4289899, upper bound: 0.4290068
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4323956, 0.4321685
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7476834, 0.7514672
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3257765, 0.3253559
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4282811, upper bound: 0.4301399
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4282811, upper bound: 0.4301399
time: 1.13 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.42 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4209202, upper bound: 0.4228556
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4209202, upper bound: 0.4228556
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4285075, upper bound: 0.4298566
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4285075, upper bound: 0.4298566
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4214865, upper bound: 0.4218887
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4214865, upper bound: 0.4218887
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4283330, upper bound: 0.4300892
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4283330, upper bound: 0.4300892
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4289899, upper bound: 0.4290068
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4289899, upper bound: 0.4290068
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4282811, upper bound: 0.4301399
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 7, lower bound: -0.4282811, upper bound: 0.4301399

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.80 + 220.64 = 224.44 seconds
