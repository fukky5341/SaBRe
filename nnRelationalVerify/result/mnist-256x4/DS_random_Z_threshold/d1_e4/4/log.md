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
execution time: IAR + RelationalAnalysis = 1.08 + 2.51 = 3.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.5042396, upper bound: 0.5042396

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788201, upper bound: 0.4788201
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788201, upper bound: 0.4788201
time: 2.26 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.57
Output dim: 7, lower bound: -0.4788201, upper bound: 0.4788201
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.57
Output dim: 7, lower bound: -0.4788201, upper bound: 0.4788201

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4350703, 0.4348806
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3311792, 0.3308300
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4468589, upper bound: 0.4468589
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4468589, upper bound: 0.4468589
time: 1.02 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4348805, 0.4351088
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3308301, 0.3312583
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4762238, upper bound: 0.4759744
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4759744, upper bound: 0.4762238
time: 1.44 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.30 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.30
Output dim: 7, lower bound: -0.4468589, upper bound: 0.4468589
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.30
Output dim: 7, lower bound: -0.4468589, upper bound: 0.4468589
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 7, lower bound: -0.4762238, upper bound: 0.4759744
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 7, lower bound: -0.4759744, upper bound: 0.4762238

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4347234, 0.4349697
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3305519, 0.3310092
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4742087, upper bound: 0.4741076
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4743574, upper bound: 0.4738768
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4347415, 0.4349516
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3305852, 0.3309759
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4337904, upper bound: 0.4338760
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4337904, upper bound: 0.4338760
time: 1.17 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.23 seconds
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 7, lower bound: -0.4742087, upper bound: 0.4741076
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 7, lower bound: -0.4743574, upper bound: 0.4738768
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 7, lower bound: -0.4337904, upper bound: 0.4338760
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 7, lower bound: -0.4337904, upper bound: 0.4338760

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4344461, 0.4346594
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3299605, 0.3303512
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4699551, upper bound: 0.4701437
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702506, upper bound: 0.4697308
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4344131, 0.4346853
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3298999, 0.3303989
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4738524, upper bound: 0.4732475
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4736455, upper bound: 0.4733628
time: 1.22 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.16 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 7, lower bound: -0.4699551, upper bound: 0.4701437
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 7, lower bound: -0.4702506, upper bound: 0.4697308
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 7, lower bound: -0.4738524, upper bound: 0.4732475
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 7, lower bound: -0.4736455, upper bound: 0.4733628

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342881, 0.4344718
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3296908, 0.3300253
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4674435, upper bound: 0.4676657
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4675079, upper bound: 0.4675791
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342585, 0.4344991
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3296360, 0.3300756
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4699865, upper bound: 0.4682145
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686100, upper bound: 0.4694642
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343925, 0.4346663
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3298638, 0.3303667
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4425623, upper bound: 0.4421225
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4425623, upper bound: 0.4421225
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4343945, 0.4346646
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3298676, 0.3303636
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4711186, upper bound: 0.4709067
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4711899, upper bound: 0.4708588
time: 1.06 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.98 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.4674435, upper bound: 0.4676657
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.4675079, upper bound: 0.4675791
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.4699865, upper bound: 0.4682145
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.4686100, upper bound: 0.4694642
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.4425623, upper bound: 0.4421225
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.4425623, upper bound: 0.4421225
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.4711186, upper bound: 0.4709067
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.4711899, upper bound: 0.4708588

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341252, 0.4343141
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3293729, 0.3297225
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4238930, upper bound: 0.4244073
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4238930, upper bound: 0.4244073
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4341305, 0.4343089
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3293828, 0.3297128
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4239007, upper bound: 0.4243998
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4239007, upper bound: 0.4243998
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4339007, 0.4342341
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3288280, 0.3294391
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4674913, upper bound: 0.4658209
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4675029, upper bound: 0.4651625
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4339997, 0.4341412
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3290102, 0.3292683
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4331566, upper bound: 0.4337191
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4331566, upper bound: 0.4337191
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342318, 0.4345059
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295551, 0.3300630
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4401136, upper bound: 0.4399629
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4401136, upper bound: 0.4399629
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4342359, 0.4345011
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3295628, 0.3300543
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4692933, upper bound: 0.4660131
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4663225, upper bound: 0.4688936
time: 0.96 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 6.66 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4238930, upper bound: 0.4244073
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4238930, upper bound: 0.4244073
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4239007, upper bound: 0.4243998
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4239007, upper bound: 0.4243998
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4674913, upper bound: 0.4658209
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4675029, upper bound: 0.4651625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4331566, upper bound: 0.4337191
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4331566, upper bound: 0.4337191
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4401136, upper bound: 0.4399629
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4401136, upper bound: 0.4399629
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4692933, upper bound: 0.4660131
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 7, lower bound: -0.4663225, upper bound: 0.4688936

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4337312, 0.4340678
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3284940, 0.3291110
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4312067, upper bound: 0.4305056
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4312067, upper bound: 0.4305056
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4337342, 0.4340566
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3284996, 0.3290905
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4312241, upper bound: 0.4302584
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4312241, upper bound: 0.4302584
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4324853, 0.4330412
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7593547, 0.7502483
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3260588, 0.3270829
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4649506, upper bound: 0.4620333
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4653051, upper bound: 0.4615618
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4327660, 0.4327511
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7545876, 0.7548589
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3265750, 0.3265492
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4379236, upper bound: 0.4386953
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4379236, upper bound: 0.4386953
time: 1.02 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.95 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.4312067, upper bound: 0.4305056
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.4312067, upper bound: 0.4305056
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.4312241, upper bound: 0.4302584
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.4312241, upper bound: 0.4302584
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.4649506, upper bound: 0.4620333
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.4653051, upper bound: 0.4615618
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.4379236, upper bound: 0.4386953
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.4379236, upper bound: 0.4386953

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4323246, 0.4328532
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7561750, 0.7475059
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3257709, 0.3267459
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4225665, upper bound: 0.4219205
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4225665, upper bound: 0.4219205
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095
1: -0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130
2: -0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546
3: -0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4322975, 0.4328757
4: -0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590
5: -0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786
6: -0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493
7: 0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7565444, 0.7470621
8: -0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3257211, 0.3267872
9: -0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4302597, upper bound: 0.4287883
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4302597, upper bound: 0.4287883
time: 0.86 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.59 seconds
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 7, lower bound: -0.4225665, upper bound: 0.4219205
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 7, lower bound: -0.4225665, upper bound: 0.4219205
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 7, lower bound: -0.4302597, upper bound: 0.4287883
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 7, lower bound: -0.4302597, upper bound: 0.4287883

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.59 + 81.81 = 85.40 seconds
