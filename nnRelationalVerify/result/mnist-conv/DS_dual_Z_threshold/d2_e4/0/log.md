## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.276117024


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6436245, 0.6436245)
1: (-2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7583723, 0.7583723)
2: (-3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8736706, 0.8736706)
3: (-12.2057600, -10.5023746, -12.2057600, -10.5023746, -1.0068245, 1.0068247)
4: (-5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8551059, 0.8551056)
5: (-2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7331014, 0.7331014)
6: (2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7932694, 0.7932692)
7: (-9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8936715, 0.8936715)
8: (-1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0782151, 1.0782149)
9: (-8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6931071, 0.6931071)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.43 + 34.42 = 56.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.2876219, upper bound: 0.2876191

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 430

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 430

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2876135, upper bound: 0.2807938
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2807969, upper bound: 0.2876153
time: 3.87 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.82 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.82
Output dim: 6, lower bound: -0.2876135, upper bound: 0.2807938
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.82
Output dim: 6, lower bound: -0.2807969, upper bound: 0.2876153

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6377511, 0.6402807
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7576437, 0.7570951
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8697987, 0.8714645
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9957213, 1.0005035
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8481941, 0.8429606
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7252204, 0.7192574
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7862163, 0.7808859
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8765521, 0.8839231
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0776472, 1.0778909
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6881056, 0.6902578

Time for backsubstitution: 21.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.47 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2791674, upper bound: 0.2781947
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2850102, upper bound: 0.2723491
time: 4.01 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6402805, 0.6377511
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7570951, 0.7576437
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8714647, 0.8697984
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -1.0005035, 0.9957211
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8429608, 0.8481939
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7192574, 0.7252207
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7808859, 0.7862160
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8839231, 0.8765521
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0778909, 1.0776470
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6902578, 0.6881056

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.44 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2723499, upper bound: 0.2850120
time: 3.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2781929, upper bound: 0.2791667
time: 3.70 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 6, lower bound: -0.2791674, upper bound: 0.2781947
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 6, lower bound: -0.2850102, upper bound: 0.2723491
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 6, lower bound: -0.2723499, upper bound: 0.2850120
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 6, lower bound: -0.2781929, upper bound: 0.2791667

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6366394, 0.6385028
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7537942, 0.7532997
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8435230, 0.8463769
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9879014, 0.9956326
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8292372, 0.8207467
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7156649, 0.7109392
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7740819, 0.7723038
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8722432, 0.8801303
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0626085, 1.0649326
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6868684, 0.6869895

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 1459

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2741765, upper bound: 0.2731453
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2741181, upper bound: 0.2732018
time: 3.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6359732, 0.6391690
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7538483, 0.7532456
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8447104, 0.8451891
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9908502, 0.9926844
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8259799, 0.8240039
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7169023, 0.7097018
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7776341, 0.7687516
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8727593, 0.8796141
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0646889, 1.0628526
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6848371, 0.6890206

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2800189, upper bound: 0.2672999
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2799604, upper bound: 0.2673583
time: 3.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6391690, 0.6359732
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7532456, 0.7538483
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8451891, 0.8447106
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9926841, 0.9908502
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8240039, 0.8259799
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7097018, 0.7169023
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7687519, 0.7776341
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8796141, 0.8727593
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0628521, 1.0646887
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6890206, 0.6848371

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1459

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2673591, upper bound: 0.2799622
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2673024, upper bound: 0.2800199
time: 3.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6385028, 0.6366394
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7532997, 0.7537942
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8463764, 0.8435228
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9956329, 0.9879017
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8207467, 0.8292370
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7109392, 0.7156649
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7723041, 0.7740819
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8801305, 0.8722432
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0649326, 1.0626085
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6869893, 0.6868684

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2732020, upper bound: 0.2741170
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2731436, upper bound: 0.2741773
time: 4.04 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.10
Output dim: 6, lower bound: -0.2741765, upper bound: 0.2731453
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.10
Output dim: 6, lower bound: -0.2741181, upper bound: 0.2732018
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.10
Output dim: 6, lower bound: -0.2800189, upper bound: 0.2672999
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.10
Output dim: 6, lower bound: -0.2799604, upper bound: 0.2673583
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.10
Output dim: 6, lower bound: -0.2673591, upper bound: 0.2799622
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.10
Output dim: 6, lower bound: -0.2673024, upper bound: 0.2800199
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.10
Output dim: 6, lower bound: -0.2732020, upper bound: 0.2741170
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.10
Output dim: 6, lower bound: -0.2731436, upper bound: 0.2741773

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6317956, 0.6333022
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7246280, 0.7253538
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8143439, 0.8206329
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9728048, 0.9861405
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8227715, 0.8131912
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.6988662, 0.6932938
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7925556, 0.7930288
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8474314, 0.8523190
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0830901, 1.0833378
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6957009, 0.6960483

Time for backsubstitution: 21.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2776485, upper bound: 0.2621164
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2745431, upper bound: 0.2649003
time: 3.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6307724, 0.6343255
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7259023, 0.7240793
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8189669, 0.8160100
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9813578, 0.9775872
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8184247, 0.8175380
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.6992568, 0.6929032
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7983592, 0.7872255
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8449481, 0.8548024
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0830939, 1.0833340
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6938961, 0.6978531

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775591, upper bound: 0.2618818
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2747761, upper bound: 0.2649894
time: 3.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6343253, 0.6307726
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7240791, 0.7259024
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8160100, 0.8189669
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9775875, 0.9813581
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8175383, 0.8184245
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.6929032, 0.6992569
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7872255, 0.7983589
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8548024, 0.8449481
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0833337, 1.0830936
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6978533, 0.6938961

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2618847, upper bound: 0.2747749
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2618847, upper bound: 0.2775578
time: 3.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6333020, 0.6317959
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7253537, 0.7246279
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8206334, 0.8143439
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9861405, 0.9728045
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8131914, 0.8227713
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.6932937, 0.6988664
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7930288, 0.7925556
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8523192, 0.8474314
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0833375, 1.0830901
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6960483, 0.6957011

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2649013, upper bound: 0.2745418
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2621174, upper bound: 0.2776471
time: 3.88 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.15 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.2776485, upper bound: 0.2621164
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.2745431, upper bound: 0.2649003
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.2775591, upper bound: 0.2618818
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.2747761, upper bound: 0.2649894
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.2618847, upper bound: 0.2747749
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.2618847, upper bound: 0.2775578
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.2649013, upper bound: 0.2745418
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.2621174, upper bound: 0.2776471

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6384001, 0.6405883
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7517965, 0.7526574
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8717229, 0.8719041
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9941344, 0.9992146
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8378696, 0.8320670
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7262011, 0.7199650
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7831264, 0.7767560
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8767934, 0.8842282
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0831439, 1.0819762
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6849644, 0.6872869

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Candidate
type: DSZ, layer: 3, pos: 1459

### Candidate
type: DSZ, layer: 3, pos: 1935

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2687197, upper bound: 0.2567574
time: 4.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2727510, upper bound: 0.2537505
time: 4.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6384001, 0.6405883
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7517965, 0.7526574
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8717229, 0.8719041
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9941344, 0.9992146
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8378696, 0.8320670
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7262011, 0.7199650
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7831264, 0.7767560
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8767934, 0.8842282
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0831439, 1.0819762
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6849644, 0.6872869

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Candidate
type: DSZ, layer: 3, pos: 1459

### Candidate
type: DSZ, layer: 3, pos: 1935

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2686302, upper bound: 0.2565239
time: 4.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2726612, upper bound: 0.2535208
time: 4.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6405883, 0.6384001
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7526574, 0.7517965
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8719041, 0.8717229
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9992146, 0.9941344
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8320670, 0.8378696
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7199650, 0.7262011
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7767560, 0.7831261
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8842285, 0.8767934
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0819762, 1.0831439
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6872869, 0.6849647

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Candidate
type: DSZ, layer: 3, pos: 1459

### Candidate
type: DSZ, layer: 3, pos: 1935

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2535208, upper bound: 0.2726601
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2565239, upper bound: 0.2686293
time: 4.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6405883, 0.6384001
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7526574, 0.7517965
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8719041, 0.8717229
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9992146, 0.9941344
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8320670, 0.8378696
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7199650, 0.7262011
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7767560, 0.7831261
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8842285, 0.8767934
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0819762, 1.0831439
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6872869, 0.6849647

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 2810

### Candidate
type: DSZ, layer: 3, pos: 1459

### Candidate
type: DSZ, layer: 3, pos: 1935

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2537530, upper bound: 0.2727500
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2567564, upper bound: 0.2687187
time: 4.05 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.41 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.41
Output dim: 6, lower bound: -0.2687197, upper bound: 0.2567574
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.41
Output dim: 6, lower bound: -0.2727510, upper bound: 0.2537505
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.41
Output dim: 6, lower bound: -0.2686302, upper bound: 0.2565239
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.41
Output dim: 6, lower bound: -0.2726612, upper bound: 0.2535208
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.41
Output dim: 6, lower bound: -0.2535208, upper bound: 0.2726601
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.41
Output dim: 6, lower bound: -0.2565239, upper bound: 0.2686293
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.41
Output dim: 6, lower bound: -0.2537530, upper bound: 0.2727500
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.41
Output dim: 6, lower bound: -0.2567564, upper bound: 0.2687187

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.85 + 428.42 = 485.27 seconds
