## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.45381564


## IAR start

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
execution time: IAR + RelationalAnalysis = 1.49 + 2.54 = 4.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.5042396, upper bound: 0.5042396

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5018988, upper bound: 0.5020359
time: 6.03 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5020359, upper bound: 0.5018988
time: 1.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.66 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.66
Output dim: 7, lower bound: -0.5018988, upper bound: 0.5020359
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.66
Output dim: 7, lower bound: -0.5020359, upper bound: 0.5018988

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4349427, 0.4349481
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3309585, 0.3309682
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4989254, upper bound: 0.4997181
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4995902, upper bound: 0.4990916
time: 2.13 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4349480, 0.4349428
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3309681, 0.3309585
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4990916, upper bound: 0.4995902
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4997181, upper bound: 0.4989254
time: 1.70 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.92 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.92
Output dim: 7, lower bound: -0.4989254, upper bound: 0.4997181
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.92
Output dim: 7, lower bound: -0.4995902, upper bound: 0.4990916
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.92
Output dim: 7, lower bound: -0.4990916, upper bound: 0.4995902
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.92
Output dim: 7, lower bound: -0.4997181, upper bound: 0.4989254

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4346541, 0.4346330
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3303032, 0.3302643
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4864121, upper bound: 0.4869442
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4862367, upper bound: 0.4871212
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4346278, 0.4346589
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3302545, 0.3303120
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4869677, upper bound: 0.4864381
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4867390, upper bound: 0.4865877
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4346589, 0.4346277
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3303120, 0.3302546
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4865877, upper bound: 0.4867390
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4864381, upper bound: 0.4869677
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4346330, 0.4346541
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3302643, 0.3303032
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4871212, upper bound: 0.4862367
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4869442, upper bound: 0.4864121
time: 1.31 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.42 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 7, lower bound: -0.4864121, upper bound: 0.4869442
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 7, lower bound: -0.4862367, upper bound: 0.4871212
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 7, lower bound: -0.4869677, upper bound: 0.4864381
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 7, lower bound: -0.4867390, upper bound: 0.4865877
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 7, lower bound: -0.4865877, upper bound: 0.4867390
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 7, lower bound: -0.4864381, upper bound: 0.4869677
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 7, lower bound: -0.4871212, upper bound: 0.4862367
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 7, lower bound: -0.4869442, upper bound: 0.4864121

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343729, 0.4344280
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3297820, 0.3298831
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4783620, upper bound: 0.4791045
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4784209, upper bound: 0.4790987
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4344459, 0.4343519
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3299159, 0.3297431
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4782649, upper bound: 0.4791954
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4782966, upper bound: 0.4791624
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343464, 0.4344519
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3297333, 0.3299272
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4789199, upper bound: 0.4785659
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4789869, upper bound: 0.4785496
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4344211, 0.4343778
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3298707, 0.3297907
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788302, upper bound: 0.4786511
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788557, upper bound: 0.4786277
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343777, 0.4344212
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3297907, 0.3298707
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4786277, upper bound: 0.4788557
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4786511, upper bound: 0.4788302
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4344518, 0.4343466
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3299272, 0.3297333
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4785496, upper bound: 0.4789869
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4785659, upper bound: 0.4789199
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343519, 0.4344458
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3297430, 0.3299159
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4791624, upper bound: 0.4782966
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4791954, upper bound: 0.4782649
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4344280, 0.4343730
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3298831, 0.3297820
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4790987, upper bound: 0.4784209
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4791045, upper bound: 0.4783620
time: 1.35 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4783620, upper bound: 0.4791045
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4784209, upper bound: 0.4790987
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4782649, upper bound: 0.4791954
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4782966, upper bound: 0.4791624
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4789199, upper bound: 0.4785659
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4789869, upper bound: 0.4785496
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4788302, upper bound: 0.4786511
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4788557, upper bound: 0.4786277
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4786277, upper bound: 0.4788557
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4786511, upper bound: 0.4788302
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4785496, upper bound: 0.4789869
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4785659, upper bound: 0.4789199
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4791624, upper bound: 0.4782966
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4791954, upper bound: 0.4782649
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4790987, upper bound: 0.4784209
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 7, lower bound: -0.4791045, upper bound: 0.4783620

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342117, 0.4342404
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295022, 0.3295551
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341855, 0.4342713
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3294540, 0.3296119
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342890, 0.4341643
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3296446, 0.3294151
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342582, 0.4341916
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295879, 0.3294654
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341853, 0.4342644
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3294536, 0.3295992
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341590, 0.4342974
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3294054, 0.3296600
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342637, 0.4341902
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295980, 0.3294628
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342337, 0.4342167
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295427, 0.3295115
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342167, 0.4342337
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295115, 0.3295427
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341903, 0.4342637
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3294627, 0.3295980
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342973, 0.4341590
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3296599, 0.3294054
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342644, 0.4341853
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295992, 0.3294537
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341915, 0.4342582
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3294653, 0.3295879
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341643, 0.4342891
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3294150, 0.3296447
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342713, 0.4341855
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3296118, 0.3294540
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342403, 0.4342117
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295551, 0.3295022
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
time: 0.89 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4132930, upper bound: 0.4136355
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4135356, upper bound: 0.4133687
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4133687, upper bound: 0.4135356
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.29
Output dim: 7, lower bound: -0.4136355, upper bound: 0.4132930

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.03 + 129.00 = 133.03 seconds
