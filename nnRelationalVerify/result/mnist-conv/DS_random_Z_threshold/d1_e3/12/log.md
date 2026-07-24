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
execution time: IAR + RelationalAnalysis = 23.06 + 32.77 = 55.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0778050, upper bound: 0.0778050

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 948

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0776352, upper bound: 0.0776913
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0776913, upper bound: 0.0776351
time: 2.94 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.88 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.88
Output dim: 7, lower bound: -0.0776352, upper bound: 0.0776913
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.88
Output dim: 7, lower bound: -0.0776913, upper bound: 0.0776351

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2673905, 0.2673357
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2110529, 0.2121371
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196793, 0.1196299
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1599371, 0.1598365
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2156794, 0.2165229
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1736912, 0.1748512
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2214214, 0.2225549
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2563238, 0.2564373
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1607304, 0.1599667
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1963724, 0.1963085

Time for backsubstitution: 7.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1976

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0774766, upper bound: 0.0775227
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0774665, upper bound: 0.0775328
time: 2.92 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2673357, 0.2673904
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2121371, 0.2110529
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196299, 0.1196793
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1598365, 0.1599371
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2165232, 0.2156794
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1748511, 0.1736910
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2225548, 0.2214212
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2564373, 0.2563238
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1599667, 0.1607304
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1963085, 0.1963724

Time for backsubstitution: 7.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1976

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0775329, upper bound: 0.0774664
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0775227, upper bound: 0.0774766
time: 3.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 13.90 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 7, lower bound: -0.0774766, upper bound: 0.0775227
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 7, lower bound: -0.0774665, upper bound: 0.0775328
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 7, lower bound: -0.0775329, upper bound: 0.0774664
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 7, lower bound: -0.0775227, upper bound: 0.0774766

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2661350, 0.2663558
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2053233, 0.2073334
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1200774, 0.1200991
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1583440, 0.1587057
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2144179, 0.2146372
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1724504, 0.1731861
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2158902, 0.2168901
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2566462, 0.2566565
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1591642, 0.1586179
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1959248, 0.1957804

Time for backsubstitution: 7.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1783

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0771986, upper bound: 0.0775143
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0774683, upper bound: 0.0772446
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2664106, 0.2660801
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2062494, 0.2064075
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1201484, 0.1200280
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1588066, 0.1582432
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2137935, 0.2152616
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1720263, 0.1736102
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2157567, 0.2170237
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2565432, 0.2567595
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1593816, 0.1584245
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1958443, 0.1958630

Time for backsubstitution: 7.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2563

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0754450, upper bound: 0.0769671
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0768963, upper bound: 0.0755369
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2660801, 0.2664106
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2064075, 0.2062494
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1200280, 0.1201485
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1582432, 0.1588066
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2152615, 0.2137936
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1736103, 0.1720262
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2170238, 0.2157565
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2567596, 0.2565430
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1584245, 0.1593816
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1958631, 0.1958442

Time for backsubstitution: 7.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1783

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0772548, upper bound: 0.0774581
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0775245, upper bound: 0.0771884
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2663558, 0.2661350
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2073334, 0.2053233
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1200991, 0.1200774
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1587057, 0.1583440
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2146373, 0.2144181
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1731862, 0.1724503
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2168901, 0.2158900
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2566566, 0.2566460
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1586179, 0.1591642
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1957804, 0.1959248

Time for backsubstitution: 8.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 1783

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2138

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0762878, upper bound: 0.0767421
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0767882, upper bound: 0.0762417
time: 3.01 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.11 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 7, lower bound: -0.0771986, upper bound: 0.0775143
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 7, lower bound: -0.0774683, upper bound: 0.0772446
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 14.11
Output dim: 7, lower bound: -0.0754450, upper bound: 0.0769671
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 14.11
Output dim: 7, lower bound: -0.0768963, upper bound: 0.0755369
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 7, lower bound: -0.0772548, upper bound: 0.0774581
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 7, lower bound: -0.0775245, upper bound: 0.0771884
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 14.11
Output dim: 7, lower bound: -0.0762878, upper bound: 0.0767421
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 14.11
Output dim: 7, lower bound: -0.0767882, upper bound: 0.0762417

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2660738, 0.2663432
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2052892, 0.2072763
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1200724, 0.1200907
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1583363, 0.1587135
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2143948, 0.2146207
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1724507, 0.1731284
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2158743, 0.2169384
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2566338, 0.2566539
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1591598, 0.1586154
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1959184, 0.1957786

Time for backsubstitution: 8.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2563

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0751753, upper bound: 0.0769486
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0766284, upper bound: 0.0755185
time: 3.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2661225, 0.2663558
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2053233, 0.2072995
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1200774, 0.1200941
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1583440, 0.1586981
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2144179, 0.2146138
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1723927, 0.1731861
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2158902, 0.2168744
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2566435, 0.2566565
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1591617, 0.1586179
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1959248, 0.1957741

Time for backsubstitution: 8.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2802

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0774626, upper bound: 0.0746962
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0750374, upper bound: 0.0772382
time: 3.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2660190, 0.2663981
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2063735, 0.2061921
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1200230, 0.1201401
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1582355, 0.1588143
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2152383, 0.2137772
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1736108, 0.1719685
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2170080, 0.2158048
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2567472, 0.2565407
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1584201, 0.1593790
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1958568, 0.1958424

Time for backsubstitution: 8.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2936

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1759

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0770672, upper bound: 0.0771564
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0769531, upper bound: 0.0772705
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2660676, 0.2664106
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2064075, 0.2062154
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1200280, 0.1201434
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1582432, 0.1587989
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2152615, 0.2137703
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1735526, 0.1720262
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2170238, 0.2157408
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2567570, 0.2565430
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1584220, 0.1593816
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1958631, 0.1958379

Time for backsubstitution: 8.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2336

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0753742, upper bound: 0.0771882
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0775244, upper bound: 0.0750380
time: 3.03 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.22
Output dim: 7, lower bound: -0.0751753, upper bound: 0.0769486
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.22
Output dim: 7, lower bound: -0.0766284, upper bound: 0.0755185
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.22
Output dim: 7, lower bound: -0.0774626, upper bound: 0.0746962
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.22
Output dim: 7, lower bound: -0.0750374, upper bound: 0.0772382
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.22
Output dim: 7, lower bound: -0.0770672, upper bound: 0.0771564
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.22
Output dim: 7, lower bound: -0.0769531, upper bound: 0.0772705
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.22
Output dim: 7, lower bound: -0.0753742, upper bound: 0.0771882
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.22
Output dim: 7, lower bound: -0.0775244, upper bound: 0.0750380

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2649297, 0.2653067
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2056258, 0.2075694
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1197980, 0.1197560
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1573727, 0.1576294
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2138468, 0.2139823
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1700715, 0.1707065
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2161502, 0.2172930
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2567267, 0.2567103
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1596652, 0.1593034
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1952598, 0.1952266

Time for backsubstitution: 8.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 436

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1507

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0773639, upper bound: 0.0746387
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0774052, upper bound: 0.0745975
time: 3.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2650734, 0.2651627
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2055930, 0.2075918
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1197393, 0.1198147
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1572757, 0.1577239
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2138821, 0.2140427
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1699129, 0.1708651
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2163094, 0.2171339
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2566972, 0.2567399
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1598788, 0.1591218
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1953567, 0.1951090

Time for backsubstitution: 8.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2563

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0750266, upper bound: 0.0767961
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0747219, upper bound: 0.0772228
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2618186, 0.2627556
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2076339, 0.2072322
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1201884, 0.1203370
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1590052, 0.1595106
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2174664, 0.2161539
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1732305, 0.1719048
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2170039, 0.2160026
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2559288, 0.2557621
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1589220, 0.1600167
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1945446, 0.1943769

Time for backsubstitution: 8.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2563

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2336

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0749170, upper bound: 0.0771563
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0770670, upper bound: 0.0750059
time: 3.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2623765, 0.2621976
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2074136, 0.2074525
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1202199, 0.1203054
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1589320, 0.1595838
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2176151, 0.2160051
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1735471, 0.1715882
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2172059, 0.2158008
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2559686, 0.2557223
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1590576, 0.1598808
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1943913, 0.1945302

Time for backsubstitution: 8.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2936

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0769460, upper bound: 0.0772026
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0768855, upper bound: 0.0772631
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2649618, 0.2652656
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2060041, 0.2054795
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1179169, 0.1178808
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1576716, 0.1574318
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2130723, 0.2130640
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1697762, 0.1690931
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2128826, 0.2123859
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2555583, 0.2556393
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1583961, 0.1593589
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1948791, 0.1938300

Time for backsubstitution: 8.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0733784, upper bound: 0.0766180
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0748083, upper bound: 0.0751664
time: 3.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2649225, 0.2653050
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2056714, 0.2058122
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1177654, 0.1180324
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1568764, 0.1582272
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2145548, 0.2115815
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1706192, 0.1682498
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2136687, 0.2115997
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2558529, 0.2553446
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1583991, 0.1593559
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1938555, 0.1948535

Time for backsubstitution: 8.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2563

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0774989, upper bound: 0.0745954
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0773726, upper bound: 0.0750226
time: 3.23 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.08 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0773639, upper bound: 0.0746387
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0774052, upper bound: 0.0745975
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0750266, upper bound: 0.0767961
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0747219, upper bound: 0.0772228
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0749170, upper bound: 0.0771563
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0770670, upper bound: 0.0750059
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0769460, upper bound: 0.0772026
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0768855, upper bound: 0.0772631
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0733784, upper bound: 0.0766180
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0748083, upper bound: 0.0751664
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0774989, upper bound: 0.0745954
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.08
Output dim: 7, lower bound: -0.0773726, upper bound: 0.0750226

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2644300, 0.2650127
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2056234, 0.2075688
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1195781, 0.1196198
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1573151, 0.1574755
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2138387, 0.2138760
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1700560, 0.1706076
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2161158, 0.2175137
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2567613, 0.2567105
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1595323, 0.1592393
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1952518, 0.1951224

Time for backsubstitution: 8.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2138

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0761289, upper bound: 0.0739043
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0766295, upper bound: 0.0734038
time: 4.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2646353, 0.2648077
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2056251, 0.2075670
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196618, 0.1195360
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1572186, 0.1575720
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2137405, 0.2139745
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1699725, 0.1706910
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2163702, 0.2172592
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2567267, 0.2567451
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1596012, 0.1591704
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1951557, 0.1952184

Time for backsubstitution: 8.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 436

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2936

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0773980, upper bound: 0.0745299
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0773375, upper bound: 0.0745903
time: 3.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2648239, 0.2651627
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2050277, 0.2075918
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196401, 0.1198147
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1572006, 0.1577239
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2136436, 0.2140427
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1699129, 0.1702828
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2162288, 0.2171339
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2566738, 0.2567399
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1598788, 0.1588478
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1953567, 0.1951061

Time for backsubstitution: 8.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2936

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0747148, upper bound: 0.0771552
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0746544, upper bound: 0.0772157
time: 3.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2607136, 0.2616112
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2072303, 0.2064960
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1180773, 0.1180743
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1584337, 0.1581439
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2152772, 0.2154472
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1694546, 0.1689720
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2128630, 0.2126478
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2547297, 0.2548578
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1588962, 0.1599938
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1935602, 0.1923692

Time for backsubstitution: 8.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0744195, upper bound: 0.0759877
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0737474, upper bound: 0.0766587
time: 3.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2606742, 0.2616507
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2068977, 0.2068287
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1179257, 0.1182259
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1576385, 0.1589390
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2167597, 0.2139649
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1702979, 0.1681290
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2136490, 0.2118616
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2550244, 0.2545631
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1588992, 0.1599908
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1925368, 0.1933926

Time for backsubstitution: 8.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 436

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2936

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0770599, upper bound: 0.0749382
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0769991, upper bound: 0.0749987
time: 4.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2623644, 0.2621903
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2073909, 0.2074382
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1202051, 0.1202791
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1589314, 0.1595834
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2175908, 0.2159896
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1735358, 0.1715802
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2171969, 0.2157971
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2559619, 0.2557220
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1590557, 0.1598763
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1943746, 0.1945102

Time for backsubstitution: 8.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 2802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0764485, upper bound: 0.0760340
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0757775, upper bound: 0.0767050
time: 3.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2623692, 0.2621855
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2073992, 0.2074299
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1201937, 0.1202905
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1589315, 0.1595834
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2175994, 0.2159810
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1735392, 0.1715769
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2172022, 0.2157919
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2559686, 0.2557154
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1590531, 0.1598790
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1943712, 0.1945136

Time for backsubstitution: 8.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0748894, upper bound: 0.0766931
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0763199, upper bound: 0.0752418
time: 3.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2654896, 0.2650558
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2061772, 0.2052671
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1180156, 0.1179482
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1570058, 0.1581521
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2138159, 0.2113426
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1700364, 0.1667020
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2138824, 0.2115191
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2558331, 0.2553213
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1581248, 0.1588662
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1938525, 0.1948485

Time for backsubstitution: 8.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2936

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0774917, upper bound: 0.0745278
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0774313, upper bound: 0.0745881
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2646732, 0.2653050
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2051263, 0.2058122
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1176811, 0.1180324
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1568015, 0.1582272
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2143157, 0.2115815
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1706192, 0.1676672
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2135882, 0.2115997
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2558298, 0.2553446
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1583991, 0.1590815
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1938555, 0.1948506

Time for backsubstitution: 8.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 436

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0773489, upper bound: 0.0741895
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0765394, upper bound: 0.0749989
time: 3.12 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 14.28 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0761289, upper bound: 0.0739043
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0766295, upper bound: 0.0734038
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0773980, upper bound: 0.0745299
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0773375, upper bound: 0.0745903
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0747148, upper bound: 0.0771552
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0746544, upper bound: 0.0772157
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0744195, upper bound: 0.0759877
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0737474, upper bound: 0.0766587
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0770599, upper bound: 0.0749382
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0769991, upper bound: 0.0749987
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0764485, upper bound: 0.0760340
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0757775, upper bound: 0.0767050
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0748894, upper bound: 0.0766931
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0763199, upper bound: 0.0752418
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0774917, upper bound: 0.0745278
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0774313, upper bound: 0.0745881
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0773489, upper bound: 0.0741895
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.28
Output dim: 7, lower bound: -0.0765394, upper bound: 0.0749989

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2646229, 0.2648002
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2056026, 0.2075529
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196469, 0.1195096
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1572181, 0.1575718
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2137166, 0.2139592
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1699618, 0.1706839
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2163614, 0.2172552
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2567198, 0.2567449
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1595993, 0.1591659
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1951391, 0.1951985

Time for backsubstitution: 8.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2563

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2138

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0761631, upper bound: 0.0737954
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0766635, upper bound: 0.0732948
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2646279, 0.2647953
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2056110, 0.2075446
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196354, 0.1195211
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1572182, 0.1575716
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2137252, 0.2139506
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1699651, 0.1706805
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2163664, 0.2172501
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2567265, 0.2567382
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1595967, 0.1591685
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1951357, 0.1952019

Time for backsubstitution: 8.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0753115, upper bound: 0.0740287
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0767674, upper bound: 0.0729512
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2648115, 0.2651554
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2050053, 0.2075775
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196251, 0.1197883
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1571999, 0.1577234
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2136197, 0.2140274
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1699020, 0.1702754
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2162194, 0.2171303
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2566669, 0.2567399
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1598769, 0.1588433
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1953402, 0.1950861

Time for backsubstitution: 8.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1759

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0745271, upper bound: 0.0768536
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0744131, upper bound: 0.0769673
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2648165, 0.2651504
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2050136, 0.2075691
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196136, 0.1197997
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1572000, 0.1577233
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2136283, 0.2140188
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1699053, 0.1702721
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2162244, 0.2171252
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2566736, 0.2567332
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1598743, 0.1588459
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1953368, 0.1950895

Time for backsubstitution: 8.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2336

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0725038, upper bound: 0.0772155
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0746542, upper bound: 0.0750652
time: 3.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2606618, 0.2616432
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2068753, 0.2068148
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1179108, 0.1181996
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1576381, 0.1589388
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2167356, 0.2139492
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1702865, 0.1681209
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2136402, 0.2118578
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2550180, 0.2545633
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1588972, 0.1599863
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1925198, 0.1933724

Time for backsubstitution: 8.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2563
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0750638, upper bound: 0.0743680
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0764942, upper bound: 0.0729167
time: 3.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2654774, 0.2650485
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2061547, 0.2052530
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1180008, 0.1179218
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1570050, 0.1581514
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2137916, 0.2113266
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1700258, 0.1666946
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2138736, 0.2115151
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2558265, 0.2553215
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1581230, 0.1588618
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1938356, 0.1948283

Time for backsubstitution: 8.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0762569, upper bound: 0.0737934
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0767572, upper bound: 0.0732918
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2654824, 0.2650437
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2061632, 0.2052445
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1179894, 0.1179332
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1570051, 0.1581513
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2138002, 0.2113183
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1700292, 0.1666913
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2138786, 0.2115101
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2558331, 0.2553146
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1581203, 0.1588644
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1938323, 0.1948316

Time for backsubstitution: 8.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 436

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0774075, upper bound: 0.0737556
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0766071, upper bound: 0.0745648
time: 3.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2646801, 0.2653176
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2051287, 0.2058147
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1176795, 0.1180311
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1567512, 0.1581722
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2142241, 0.2115002
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1706048, 0.1676599
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2135215, 0.2115399
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2556486, 0.2551312
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1583506, 0.1590396
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1939276, 0.1949139

Time for backsubstitution: 8.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1759
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 654

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1759

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0771602, upper bound: 0.0737961
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0770454, upper bound: 0.0740353
time: 3.21 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 14.58 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0761631, upper bound: 0.0737954
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0766635, upper bound: 0.0732948
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0753115, upper bound: 0.0740287
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0767674, upper bound: 0.0729512
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0745271, upper bound: 0.0768536
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0744131, upper bound: 0.0769673
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0725038, upper bound: 0.0772155
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0746542, upper bound: 0.0750652
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0750638, upper bound: 0.0743680
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0764942, upper bound: 0.0729167
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0762569, upper bound: 0.0737934
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0767572, upper bound: 0.0732918
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0774075, upper bound: 0.0737556
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0766071, upper bound: 0.0745648
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0771602, upper bound: 0.0737961
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 14.58
Output dim: 7, lower bound: -0.0770454, upper bound: 0.0740353

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2637110, 0.2640055
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2046102, 0.2068334
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1175025, 0.1175369
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1566284, 0.1563567
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2114382, 0.2133117
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1661291, 0.1673386
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2120837, 0.2137700
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2554748, 0.2558297
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1598487, 0.1588230
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1943526, 0.1930815

Time for backsubstitution: 8.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 436
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1507

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724054, upper bound: 0.0771581
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724466, upper bound: 0.0771169
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2654891, 0.2650559
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2061658, 0.2052468
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1179878, 0.1179320
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1569550, 0.1580966
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2137086, 0.2112374
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1700147, 0.1666839
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2138116, 0.2114503
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2556520, 0.2551012
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1580720, 0.1588223
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1939048, 0.1948949

Time for backsubstitution: 8.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0754117, upper bound: 0.0731854
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0768419, upper bound: 0.0717659
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2604849, 0.2616804
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2063888, 0.2068540
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1178449, 0.1182281
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1575097, 0.1588572
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2164834, 0.2139080
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1702602, 0.1676294
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2135378, 0.2117584
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2548294, 0.2543526
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1588609, 0.1596854
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1926154, 0.1934485

Time for backsubstitution: 8.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 654

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2936

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0771534, upper bound: 0.0737397
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0770924, upper bound: 0.0737896
time: 3.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2610428, 0.2611223
1: -6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2061685, 0.2070744
2: -3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1178765, 0.1181965
3: -4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1574436, 0.1589305
4: -7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2166321, 0.2137647
5: -9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1705770, 0.1673151
6: -12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2137398, 0.2115564
7: 3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2548695, 0.2543128
8: -1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1589966, 0.1595497
9: -1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.1924621, 0.1936018

Time for backsubstitution: 8.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2802
type: DSZ, layer: 3, pos: 2936
type: DSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0750495, upper bound: 0.0734651
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0764798, upper bound: 0.0720122
time: 3.11 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 15.00 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 15.00
Output dim: 7, lower bound: -0.0724054, upper bound: 0.0771581
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 15.00
Output dim: 7, lower bound: -0.0724466, upper bound: 0.0771169
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 15.00
Output dim: 7, lower bound: -0.0754117, upper bound: 0.0731854
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 15.00
Output dim: 7, lower bound: -0.0768419, upper bound: 0.0717659
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 15.00
Output dim: 7, lower bound: -0.0771534, upper bound: 0.0737397
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 15.00
Output dim: 7, lower bound: -0.0770924, upper bound: 0.0737896
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 15.00
Output dim: 7, lower bound: -0.0750495, upper bound: 0.0734651
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 15.00
Output dim: 7, lower bound: -0.0764798, upper bound: 0.0720122

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.82 + 550.35 = 606.17 seconds
