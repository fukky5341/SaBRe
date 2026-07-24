## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.07702695


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2667882, 0.2667881)
1: (-6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2112004, 0.2112004)
2: (-3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196281, 0.1196281)
3: (-4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1597021, 0.1597021)
4: (-7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2205417, 0.2205417)
5: (-9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1746140, 0.1746137)
6: (-12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2252733, 0.2252733)
7: (3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2558870, 0.2558873)
8: (-1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1682277, 0.1682276)
9: (-1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.2075396, 0.2075393)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.29 + 34.09 = 56.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0778050, upper bound: 0.0778050

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2563

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0758427, upper bound: 0.0772697
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0772698, upper bound: 0.0758427
time: 2.99 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.46
Output dim: 7, lower bound: -0.0758427, upper bound: 0.0772697
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.46
Output dim: 7, lower bound: -0.0772698, upper bound: 0.0758427

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2667232, 0.2663777
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2084373, 0.2083802
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1145765, 0.1148645
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1563334, 0.1568304
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2056847, 0.2071695
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1548808, 0.1542035
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2207367, 0.2208149
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2557828, 0.2559323
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1636851, 0.1632541
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1987450, 0.1984228

Time for backsubstitution: 8.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2563

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 1759

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0756551, upper bound: 0.0769681
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0755410, upper bound: 0.0770821
time: 3.15 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2663777, 0.2667233
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2083802, 0.2084373
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1148645, 0.1145765
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1568304, 0.1563334
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2071695, 0.2056847
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1542035, 0.1548808
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2208149, 0.2207369
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2559323, 0.2557828
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1632540, 0.1636852
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1984227, 0.1987451

Time for backsubstitution: 8.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2563

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1759

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0770822, upper bound: 0.0755410
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0769681, upper bound: 0.0756550
time: 3.06 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.93 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 14.93
Output dim: 7, lower bound: -0.0756551, upper bound: 0.0769681
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.93
Output dim: 7, lower bound: -0.0755410, upper bound: 0.0770821
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.93
Output dim: 7, lower bound: -0.0770822, upper bound: 0.0755410
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 14.93
Output dim: 7, lower bound: -0.0769681, upper bound: 0.0756550

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2630808, 0.2621775
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2094773, 0.2096406
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1147735, 0.1150299
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1570297, 0.1576002
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2080619, 0.2093982
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1548177, 0.1538239
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2209350, 0.2208114
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2550044, 0.2551141
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1643229, 0.1637559
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1972799, 0.1971109

Time for backsubstitution: 7.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2563

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0750435, upper bound: 0.0759135
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0743724, upper bound: 0.0765846
time: 3.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2621775, 0.2630810
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2096406, 0.2094773
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1150299, 0.1147735
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1576000, 0.1570299
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2093980, 0.2080618
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1538239, 0.1548177
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2208112, 0.2209351
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2551141, 0.2550046
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1637559, 0.1643226
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1971109, 0.1972799

Time for backsubstitution: 8.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2563

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0765846, upper bound: 0.0743724
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0759136, upper bound: 0.0750435
time: 3.01 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 15.01 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 15.01
Output dim: 7, lower bound: -0.0750435, upper bound: 0.0759135
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 15.01
Output dim: 7, lower bound: -0.0743724, upper bound: 0.0765846
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 15.01
Output dim: 7, lower bound: -0.0765846, upper bound: 0.0743724
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 15.01
Output dim: 7, lower bound: -0.0759136, upper bound: 0.0750435

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.38 + 66.12 = 122.50 seconds
