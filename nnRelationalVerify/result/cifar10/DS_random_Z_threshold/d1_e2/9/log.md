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
execution time: IAR + RelationalAnalysis = 8.27 + 26.26 = 34.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0162988, upper bound: 0.0163024

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2069

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162907, upper bound: 0.0162969
time: 127.69 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162924, upper bound: 0.0162925
time: 201.02 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 328.73 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 328.73
Output dim: 5, lower bound: -0.0162907, upper bound: 0.0162969
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 328.73
Output dim: 5, lower bound: -0.0162924, upper bound: 0.0162925

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1205663, 0.1205516
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2033406, 0.2033418
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324368, 0.0324371
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413689, 0.0413688
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042522, 0.1042530
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403312, 0.0403312
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376038, 0.0376035
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864249, 0.0864254
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1663735, 0.1663320
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437124, 0.2437148

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 499

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 767

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162866, upper bound: 0.0162935
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162865, upper bound: 0.0162885
time: 241.40 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1205516, 0.1205663
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2033419, 0.2033405
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324371, 0.0324368
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413688, 0.0413689
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042530, 0.1042522
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403312, 0.0403312
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376035, 0.0376038
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864254, 0.0864249
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1663319, 0.1663735
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437148, 0.2437123

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2549

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162924, upper bound: 0.0162955
time: 15.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162924, upper bound: 0.0162927
time: 19.07 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 41.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 41.35
Output dim: 5, lower bound: -0.0162866, upper bound: 0.0162935
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 41.35
Output dim: 5, lower bound: -0.0162865, upper bound: 0.0162885
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 41.35
Output dim: 5, lower bound: -0.0162924, upper bound: 0.0162955
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 41.35
Output dim: 5, lower bound: -0.0162924, upper bound: 0.0162927

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1203967, 0.1203150
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2030130, 0.2029431
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324350, 0.0324326
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413681, 0.0413679
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042512, 0.1042520
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403268, 0.0403269
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375995, 0.0375973
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864198, 0.0864215
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1661691, 0.1660542
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437305, 0.2436704

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2652

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2922

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162867, upper bound: 0.0162937
time: 5.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162867, upper bound: 0.0162935
time: 6.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1203297, 0.1203820
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2029418, 0.2030143
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324323, 0.0324354
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413681, 0.0413679
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042512, 0.1042520
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403270, 0.0403268
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375975, 0.0375992
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864209, 0.0864204
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1660957, 0.1661276
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2436680, 0.2437330

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3018

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 458

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162865, upper bound: 0.0162843
time: 5.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162775, upper bound: 0.0162911
time: 10.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1205516, 0.1205663
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2033419, 0.2033405
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324371, 0.0324368
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413688, 0.0413689
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042530, 0.1042522
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403312, 0.0403312
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376035, 0.0376038
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864254, 0.0864249
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1663319, 0.1663735
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437148, 0.2437123

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 840

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162925, upper bound: 0.0162946
time: 73.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162925, upper bound: 0.0162947
time: 78.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1205516, 0.1205663
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2033419, 0.2033405
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324371, 0.0324368
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413688, 0.0413689
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042530, 0.1042522
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403312, 0.0403312
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376035, 0.0376038
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864254, 0.0864249
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1663319, 0.1663735
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437148, 0.2437123

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2285

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2443

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162902, upper bound: 0.0162951
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162912, upper bound: 0.0162944
time: 4.05 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 15.49 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 5, lower bound: -0.0162867, upper bound: 0.0162937
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 5, lower bound: -0.0162867, upper bound: 0.0162935
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 15.49
Output dim: 5, lower bound: -0.0162865, upper bound: 0.0162843
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 5, lower bound: -0.0162775, upper bound: 0.0162911
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 5, lower bound: -0.0162925, upper bound: 0.0162946
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 5, lower bound: -0.0162925, upper bound: 0.0162947
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 5, lower bound: -0.0162902, upper bound: 0.0162951
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 5, lower bound: -0.0162912, upper bound: 0.0162944

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1203967, 0.1203150
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2030130, 0.2029431
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324350, 0.0324326
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413681, 0.0413679
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042512, 0.1042520
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403268, 0.0403269
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375995, 0.0375973
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864198, 0.0864215
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1661691, 0.1660542
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437305, 0.2436704

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 737

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 740

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162862, upper bound: 0.0162937
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162861, upper bound: 0.0162919
time: 17.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1203967, 0.1203150
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2030130, 0.2029431
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324350, 0.0324326
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413681, 0.0413679
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042512, 0.1042520
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403268, 0.0403269
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375995, 0.0375973
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864198, 0.0864215
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1661691, 0.1660542
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437305, 0.2436704

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2658

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162863, upper bound: 0.0162932
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162858, upper bound: 0.0162901
time: 89.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1199103, 0.1199388
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2028555, 0.2029260
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324163, 0.0324188
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413117, 0.0413155
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1040692, 0.1040599
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402343, 0.0402389
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375389, 0.0375378
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864119, 0.0864123
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1660183, 0.1660455
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2436060, 0.2436736

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2357

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2443

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162752, upper bound: 0.0162881
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162761, upper bound: 0.0162871
time: 87.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1205516, 0.1205663
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2033419, 0.2033405
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324371, 0.0324368
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413688, 0.0413689
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042530, 0.1042522
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403312, 0.0403312
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376035, 0.0376038
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864254, 0.0864249
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1663319, 0.1663735
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437148, 0.2437123

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162912, upper bound: 0.0162932
time: 175.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162922, upper bound: 0.0162932
time: 111.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1205516, 0.1205663
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2033419, 0.2033405
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324371, 0.0324368
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413688, 0.0413689
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042530, 0.1042522
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403312, 0.0403312
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376035, 0.0376038
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864254, 0.0864249
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1663319, 0.1663735
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437148, 0.2437123

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2264

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2585

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162860, upper bound: 0.0162861
time: 71.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162840, upper bound: 0.0162888
time: 88.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1197296, 0.1197333
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2013762, 0.2013471
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324119, 0.0324110
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412525, 0.0412479
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042212, 0.1042209
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402739, 0.0402754
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375881, 0.0375885
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0861503, 0.0861601
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1626960, 0.1626402
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2405701, 0.2404779

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 746

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3521

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162863, upper bound: 0.0162888
time: 80.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162887, upper bound: 0.0162897
time: 4.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1197186, 0.1197443
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2013484, 0.2013749
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324113, 0.0324115
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412477, 0.0412526
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042218, 0.1042204
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402754, 0.0402739
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375882, 0.0375884
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0861607, 0.0861497
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1625987, 0.1627375
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2404803, 0.2405677

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3521

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162871, upper bound: 0.0162922
time: 17.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162904, upper bound: 0.0162860
time: 40.37 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 63.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162862, upper bound: 0.0162937
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162861, upper bound: 0.0162919
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162863, upper bound: 0.0162932
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162858, upper bound: 0.0162901
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162752, upper bound: 0.0162881
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162761, upper bound: 0.0162871
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162912, upper bound: 0.0162932
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162922, upper bound: 0.0162932
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162860, upper bound: 0.0162861
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162840, upper bound: 0.0162888
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162863, upper bound: 0.0162888
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162887, upper bound: 0.0162897
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162871, upper bound: 0.0162922
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 63.99
Output dim: 5, lower bound: -0.0162904, upper bound: 0.0162860

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1203291, 0.1202481
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2028949, 0.2028289
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324342, 0.0324321
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413680, 0.0413679
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042514, 0.1042522
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403266, 0.0403266
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375978, 0.0375956
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864198, 0.0864214
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1661132, 0.1659989
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437064, 0.2436484

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 791

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3018

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162829, upper bound: 0.0162891
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162835, upper bound: 0.0162879
time: 16.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1203299, 0.1202473
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2028988, 0.2028250
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324345, 0.0324318
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413680, 0.0413679
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042514, 0.1042521
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403265, 0.0403266
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375978, 0.0375956
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864198, 0.0864214
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1661138, 0.1659984
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437085, 0.2436463

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162773, upper bound: 0.0162843
time: 88.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162812, upper bound: 0.0162849
time: 4.75 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 99.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 99.70
Output dim: 5, lower bound: -0.0162829, upper bound: 0.0162891
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 99.70
Output dim: 5, lower bound: -0.0162835, upper bound: 0.0162879
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 99.70
Output dim: 5, lower bound: -0.0162773, upper bound: 0.0162843
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 99.70
Output dim: 5, lower bound: -0.0162812, upper bound: 0.0162849
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162863, upper bound: 0.0162932
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162858, upper bound: 0.0162901
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162752, upper bound: 0.0162881
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162912, upper bound: 0.0162932
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162922, upper bound: 0.0162932
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162840, upper bound: 0.0162888
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162863, upper bound: 0.0162888
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162887, upper bound: 0.0162897
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162871, upper bound: 0.0162922
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 5, lower bound: -0.0162904, upper bound: 0.0162860

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 34.53 + 1804.59 = 1839.12 seconds
