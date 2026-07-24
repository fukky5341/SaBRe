## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 13)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0161819712


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3926678, 0.3926679)
1: (-0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0668693, 0.0668693)
2: (-2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2649534, 0.2649534)
3: (-2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4077482, 0.4077482)
4: (-3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4192618, 0.4192618)
5: (-2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2147810, 0.2147810)
6: (-2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1985283, 0.1985283)
7: (-2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5125906, 0.5125906)
8: (-0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476350, 0.1476350)
9: (-0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2112329, 0.2112329)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.88 + 28.51 = 36.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0162071, upper bound: 0.0162100

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 897

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2500

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161714, upper bound: 0.0161710
time: 41.11 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161718, upper bound: 0.0161891
time: 3.63 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 44.75 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 44.75
Output dim: 1, lower bound: -0.0161714, upper bound: 0.0161710
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 44.75
Output dim: 1, lower bound: -0.0161718, upper bound: 0.0161891

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3926625, 0.3926624
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0652785, 0.0652656
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2643810, 0.2643804
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4077044, 0.4077047
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4191037, 0.4190996
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2147685, 0.2147684
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1983718, 0.1983692
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5125663, 0.5125670
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476251, 0.1476251
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2108099, 0.2108035

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2835

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2862

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161685, upper bound: 0.0161710
time: 75.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161685, upper bound: 0.0161715
time: 52.87 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 134.39 seconds
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 134.39
Output dim: 1, lower bound: -0.0161685, upper bound: 0.0161710
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 134.39
Output dim: 1, lower bound: -0.0161685, upper bound: 0.0161715

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 36.39 + 179.14 = 215.53 seconds
