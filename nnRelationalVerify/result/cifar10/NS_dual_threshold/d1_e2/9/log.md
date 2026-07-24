## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 9)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0162870966


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1206218, 0.1206218)
1: (-2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2034255, 0.2034255)
2: (-0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324391, 0.0324391)
3: (0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413695, 0.0413695)
4: (-0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042539, 0.1042539)
5: (0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403319, 0.0403319)
6: (-0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376193, 0.0376193)
7: (-0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864264, 0.0864264)
8: (-3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1665304, 0.1665304)
9: (-0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437801, 0.2437800)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.15 + 26.17 = 33.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0162988, upper bound: 0.0163024

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 387
type: A, layer: 1, pos: 387
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2357
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2615
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2406
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2351
type: B, layer: 1, pos: 2351
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2585
type: B, layer: 1, pos: 2585
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2108
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 2096
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2453
type: A, layer: 1, pos: 2453
type: B, layer: 1, pos: 2359
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3132
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 2332
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2443
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 2285
type: A, layer: 1, pos: 2285
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 3521
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 3233
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 3105
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2888
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2431
type: B, layer: 1, pos: 2431
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 499
type: B, layer: 1, pos: 499
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2910
type: A, layer: 1, pos: 2915
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3369
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2910
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3369

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3063

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162753, upper bound: 0.0162771
time: 3.95 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162753, upper bound: 0.0162752
time: 63.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 67.92 seconds
NS_B1, status: Status.VERIFIED, split count: 1, time: 67.92
Output dim: 5, lower bound: -0.0162753, upper bound: 0.0162771
NS_B2, status: Status.VERIFIED, split count: 1, time: 67.92
Output dim: 5, lower bound: -0.0162753, upper bound: 0.0162752

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 33.32 + 67.92 = 101.24 seconds
