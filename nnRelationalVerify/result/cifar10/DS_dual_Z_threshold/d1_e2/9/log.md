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
execution time: IAR + RelationalAnalysis = 7.15 + 26.49 = 33.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0162988, upper bound: 0.0163024

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3021

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162968, upper bound: 0.0163029
time: 23.02 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162986, upper bound: 0.0163023
time: 33.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 56.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 56.59
Output dim: 5, lower bound: -0.0162968, upper bound: 0.0163029
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 56.59
Output dim: 5, lower bound: -0.0162986, upper bound: 0.0163023

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1206303, 0.1206301
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2034305, 0.2034326
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324379, 0.0324379
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413709, 0.0413710
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042527, 0.1042527
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403319, 0.0403319
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376016, 0.0376019
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864075, 0.0864068
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1665297, 0.1665303
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437643, 0.2437645

Time for backsubstitution: 5.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2453

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162935, upper bound: 0.0162959
time: 127.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162953, upper bound: 0.0162989
time: 10.10 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1206301, 0.1206303
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2034326, 0.2034305
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324379, 0.0324379
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413710, 0.0413709
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042527, 0.1042527
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403319, 0.0403319
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376019, 0.0376016
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864068, 0.0864075
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1665302, 0.1665297
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437645, 0.2437643

Time for backsubstitution: 5.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2453

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162944, upper bound: 0.0163017
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162965, upper bound: 0.0162963
time: 250.25 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 259.54 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 259.54
Output dim: 5, lower bound: -0.0162935, upper bound: 0.0162959
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 259.54
Output dim: 5, lower bound: -0.0162953, upper bound: 0.0162989
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 259.54
Output dim: 5, lower bound: -0.0162944, upper bound: 0.0163017
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 259.54
Output dim: 5, lower bound: -0.0162965, upper bound: 0.0162963

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1197904, 0.1197570
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2008678, 0.2007895
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0323868, 0.0323854
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412969, 0.0412950
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042176, 0.1042183
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403042, 0.0403048
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375944, 0.0375947
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862431, 0.0862475
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1645692, 0.1645139
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2416402, 0.2415743

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162884, upper bound: 0.0162970
time: 23.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162907, upper bound: 0.0162976
time: 4.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1197571, 0.1197902
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2007874, 0.2008698
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0323854, 0.0323868
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412950, 0.0412970
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042183, 0.1042176
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403048, 0.0403042
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375944, 0.0375947
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862482, 0.0862423
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1645134, 0.1645697
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2415741, 0.2416404

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162902, upper bound: 0.0162938
time: 82.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162936, upper bound: 0.0162935
time: 34.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1197902, 0.1197571
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2008698, 0.2007874
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0323868, 0.0323854
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412970, 0.0412950
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042176, 0.1042183
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403042, 0.0403048
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375947, 0.0375944
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862423, 0.0862482
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1645697, 0.1645134
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2416405, 0.2415741

Time for backsubstitution: 5.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162893, upper bound: 0.0162977
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162922, upper bound: 0.0162948
time: 23.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1197570, 0.1197904
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2007895, 0.2008678
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0323854, 0.0323868
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412950, 0.0412969
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042183, 0.1042176
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403048, 0.0403042
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375947, 0.0375944
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862475, 0.0862431
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1645139, 0.1645693
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2415743, 0.2416402

Time for backsubstitution: 5.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162917, upper bound: 0.0162959
time: 14.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162944, upper bound: 0.0162938
time: 3.80 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 23.56 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 23.56
Output dim: 5, lower bound: -0.0162884, upper bound: 0.0162970
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 23.56
Output dim: 5, lower bound: -0.0162907, upper bound: 0.0162976
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 23.56
Output dim: 5, lower bound: -0.0162902, upper bound: 0.0162938
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 23.56
Output dim: 5, lower bound: -0.0162936, upper bound: 0.0162935
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 23.56
Output dim: 5, lower bound: -0.0162893, upper bound: 0.0162977
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 23.56
Output dim: 5, lower bound: -0.0162922, upper bound: 0.0162948
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 23.56
Output dim: 5, lower bound: -0.0162917, upper bound: 0.0162959
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 23.56
Output dim: 5, lower bound: -0.0162944, upper bound: 0.0162938

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1155687, 0.1155001
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1933681, 0.1931884
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322508, 0.0322541
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412998, 0.0412981
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042189, 0.1042197
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402550, 0.0402557
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375774, 0.0375781
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862366, 0.0862394
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1600934, 0.1599995
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2379957, 0.2378530

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162855, upper bound: 0.0162985
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162872, upper bound: 0.0162965
time: 4.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1155335, 0.1155353
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1932667, 0.1932898
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322556, 0.0322493
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412999, 0.0412979
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042190, 0.1042196
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402550, 0.0402556
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375779, 0.0375777
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862350, 0.0862410
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1600548, 0.1600381
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2379189, 0.2379298

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162878, upper bound: 0.0162958
time: 36.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162903, upper bound: 0.0162945
time: 4.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1155355, 0.1155333
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1932877, 0.1932688
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322493, 0.0322555
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412979, 0.0413000
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042196, 0.1042190
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402556, 0.0402550
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375774, 0.0375782
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862417, 0.0862343
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1600376, 0.1600553
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2379296, 0.2379191

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162881, upper bound: 0.0162962
time: 4.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162902, upper bound: 0.0162930
time: 39.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1155003, 0.1155685
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1931864, 0.1933701
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322541, 0.0322507
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412980, 0.0412998
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042197, 0.1042189
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402557, 0.0402550
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375778, 0.0375777
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862401, 0.0862358
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1599990, 0.1600938
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2378527, 0.2379959

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162906, upper bound: 0.0162885
time: 123.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162928, upper bound: 0.0162894
time: 30.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1155685, 0.1155003
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1933701, 0.1931864
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322507, 0.0322541
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412998, 0.0412980
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042189, 0.1042197
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402550, 0.0402557
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375777, 0.0375778
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862358, 0.0862401
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1600939, 0.1599990
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2379960, 0.2378528

Time for backsubstitution: 5.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162864, upper bound: 0.0162980
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162882, upper bound: 0.0162908
time: 150.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1155334, 0.1155355
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1932688, 0.1932877
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322555, 0.0322493
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413000, 0.0412979
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042190, 0.1042196
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402550, 0.0402556
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375782, 0.0375774
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862343, 0.0862417
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1600553, 0.1600376
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2379191, 0.2379296

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162886, upper bound: 0.0162945
time: 18.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162908, upper bound: 0.0162923
time: 6.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1155353, 0.1155335
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1932897, 0.1932667
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322493, 0.0322556
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412979, 0.0412999
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042196, 0.1042190
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402556, 0.0402550
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375777, 0.0375779
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862410, 0.0862350
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1600380, 0.1600548
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2379298, 0.2379189

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162890, upper bound: 0.0162862
time: 115.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162911, upper bound: 0.0162889
time: 18.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1155001, 0.1155687
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1931884, 0.1933681
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322541, 0.0322508
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412981, 0.0412998
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042197, 0.1042189
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402557, 0.0402550
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375781, 0.0375774
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862394, 0.0862366
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1599995, 0.1600934
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2378530, 0.2379957

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162916, upper bound: 0.0162929
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162931, upper bound: 0.0162897
time: 20.76 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162855, upper bound: 0.0162985
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162872, upper bound: 0.0162965
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162878, upper bound: 0.0162958
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162903, upper bound: 0.0162945
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162881, upper bound: 0.0162962
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162902, upper bound: 0.0162930
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162906, upper bound: 0.0162885
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162928, upper bound: 0.0162894
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162864, upper bound: 0.0162980
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162882, upper bound: 0.0162908
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162886, upper bound: 0.0162945
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162908, upper bound: 0.0162923
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162890, upper bound: 0.0162862
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162911, upper bound: 0.0162889
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162916, upper bound: 0.0162929
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.47
Output dim: 5, lower bound: -0.0162931, upper bound: 0.0162897

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1151337, 0.1150117
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1926007, 0.1923732
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322388, 0.0322393
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412960, 0.0412946
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042161, 0.1042169
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402486, 0.0402495
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375733, 0.0375740
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862345, 0.0862375
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1597330, 0.1596017
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2377090, 0.2375444

Time for backsubstitution: 5.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162788, upper bound: 0.0162860
time: 68.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162775, upper bound: 0.0162929
time: 4.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1150803, 0.1150651
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1925529, 0.1924211
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322360, 0.0322421
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412963, 0.0412943
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042161, 0.1042169
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402488, 0.0402492
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375734, 0.0375740
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862346, 0.0862373
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1596956, 0.1596392
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2376871, 0.2375663

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162810, upper bound: 0.0162844
time: 21.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162792, upper bound: 0.0162906
time: 7.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1150985, 0.1150469
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1924994, 0.1924745
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322436, 0.0322345
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412961, 0.0412945
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042162, 0.1042168
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402486, 0.0402494
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375738, 0.0375736
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862329, 0.0862390
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1596945, 0.1596403
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2376322, 0.2376212

Time for backsubstitution: 5.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162814, upper bound: 0.0162860
time: 37.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162792, upper bound: 0.0162901
time: 4.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1150451, 0.1151003
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1924515, 0.1925224
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322407, 0.0322373
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412965, 0.0412941
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042162, 0.1042168
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402488, 0.0402492
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375738, 0.0375736
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862330, 0.0862389
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1596571, 0.1596777
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2376103, 0.2376431

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162835, upper bound: 0.0162831
time: 116.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162813, upper bound: 0.0162870
time: 20.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1151005, 0.1150449
1: -2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.1925203, 0.1924535
2: -0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0322373, 0.0322407
3: 0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0412941, 0.0412965
4: -0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042168, 0.1042162
5: 0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0402492, 0.0402488
6: -0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0375733, 0.0375741
7: -0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0862397, 0.0862323
8: -3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1596772, 0.1596576
9: -0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2376429, 0.2376105

Time for backsubstitution: 5.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3521

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162811, upper bound: 0.0162856
time: 21.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162802, upper bound: 0.0162884
time: 147.10 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 174.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162788, upper bound: 0.0162860
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162775, upper bound: 0.0162929
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162810, upper bound: 0.0162844
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162792, upper bound: 0.0162906
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162814, upper bound: 0.0162860
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162792, upper bound: 0.0162901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162835, upper bound: 0.0162831
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162813, upper bound: 0.0162870
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162811, upper bound: 0.0162856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 174.33
Output dim: 5, lower bound: -0.0162802, upper bound: 0.0162884
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162902, upper bound: 0.0162930
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162906, upper bound: 0.0162885
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162928, upper bound: 0.0162894
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162864, upper bound: 0.0162980
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162882, upper bound: 0.0162908
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162886, upper bound: 0.0162945
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162908, upper bound: 0.0162923
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162890, upper bound: 0.0162862
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162911, upper bound: 0.0162889
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162916, upper bound: 0.0162929
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 174.33
Output dim: 5, lower bound: -0.0162931, upper bound: 0.0162897

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 33.64 + 1779.59 = 1813.23 seconds
